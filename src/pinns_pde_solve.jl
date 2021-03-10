import Base.Broadcast
Base.Broadcast.dottable(x::Function) = true
RuntimeGeneratedFunctions.init(@__MODULE__)
"""
Algorithm for solving Physics-Informed Neural Networks problems.

Arguments:
* `chain`: a Flux.jl chain with a d-dimensional input and a 1-dimensional output,
* `strategy`: determines which training strategy will be used,
* `init_params`: the initial parameter of the neural network,
* `phi`: a trial solution,
* `derivative`: method that calculates the derivative.

"""
abstract type AbstractPINN{isinplace} <: SciMLBase.SciMLProblem end

struct PhysicsInformedNN{isinplace,C,T,P,PH,DER,K} <: AbstractPINN{isinplace}
  chain::C
  strategy::T
  init_params::P
  phi::PH
  derivative::DER
  kwargs::K

 @add_kwonly function PhysicsInformedNN{iip}(chain,
                                             strategy;
                                             init_params = nothing,
                                             phi = nothing,
                                             derivative = nothing,
                                             kwargs...) where iip
        if init_params == nothing
            if chain isa AbstractArray
                initθ = DiffEqFlux.initial_params.(chain)
            else
                initθ = DiffEqFlux.initial_params(chain)
            end

        else
            initθ = init_params
        end

        if phi == nothing
            if chain isa AbstractArray
                _phi = get_phi.(chain)
            else
                _phi = get_phi(chain)
            end
        else
            _phi = phi
        end

        if derivative == nothing
            if strategy isa QuadratureTraining
                _derivative = get_numeric_derivative(;vectorize = false)
            else
                _derivative = get_numeric_derivative(;vectorize = true)
            end
        else
            _derivative = derivative
        end
        new{iip,typeof(chain),typeof(strategy),typeof(initθ),typeof(_phi),typeof(_derivative),typeof(kwargs)}(chain,strategy,initθ,_phi,_derivative, kwargs)
    end
end
PhysicsInformedNN(chain,strategy,args...;kwargs...) = PhysicsInformedNN{true}(chain,strategy,args...;kwargs...)

SciMLBase.isinplace(prob::PhysicsInformedNN{iip}) where iip = iip


abstract type TrainingStrategies  end

"""
* `dx`: the discretization of the grid.
"""
struct GridTraining <: TrainingStrategies
    dx
end

"""
* `points`: number of points in random select training set.
"""
struct StochasticTraining <:TrainingStrategies
    points:: Int64
end

"""
* `points`:  the number of quasi-random points in minibatch,
* `sampling_alg`: the quasi-Monte Carlo sampling algorithm,
* `minibatch`: the number of subsets.

For more information look: QuasiMonteCarlo.jl https://github.com/SciML/QuasiMonteCarlo.jl
"""
struct QuasiRandomTraining <:TrainingStrategies
    points:: Int64
    sampling_alg::QuasiMonteCarlo.SamplingAlgorithm
    minibatch:: Int64
end
function QuasiRandomTraining(points;sampling_alg = UniformSample(),minibatch=500)
    QuasiRandomTraining(points,sampling_alg,minibatch)
end
"""
* `quadrature_alg`: quadrature algorithm,
* `reltol`: relative tolerance,
* `abstol`: absolute tolerance,
* `maxiters`: the maximum number of iterations in quadrature algorithm,
* `batch`: the preferred number of points to batch.

For more information look: Quadrature.jl https://github.com/SciML/Quadrature.jl
"""
struct QuadratureTraining <: TrainingStrategies
    quadrature_alg::DiffEqBase.AbstractQuadratureAlgorithm
    reltol::Float64
    abstol::Float64
    maxiters::Int64
    batch::Int64
end

function QuadratureTraining(;quadrature_alg=HCubatureJL(),reltol= 1e-6,abstol= 1e-3,maxiters=1e3,batch=0)
    QuadratureTraining(quadrature_alg,reltol,abstol,maxiters,batch)
end

"""
Create dictionary: variable => unique number for variable

# Example 1

Dict{Symbol,Int64} with 3 entries:
  :y => 2
  :t => 3
  :x => 1

# Example 2

 Dict{Symbol,Int64} with 2 entries:
  :u1 => 1
  :u2 => 2
"""
get_dict_vars(vars) = Dict( [Symbol(v) .=> i for (i,v) in enumerate(vars)])

# Wrapper for _transform_expression
function transform_expression(ex,dict_indvars,dict_depvars, initθ, strategy)
    if ex isa Expr
        ex = _transform_expression(ex,dict_indvars,dict_depvars,initθ,strategy)
    end
    return ex
end

function get_ε(dim, der_num)
    epsilon = cbrt(eps(Float32))
    ε = zeros(Float32, dim)
    ε[der_num] = epsilon
    ε
end

θ = gensym("θ")

"""
Transform the derivative expression to inner representation

# Examples

1. First compute the derivative of function 'u(x,y)' with respect to x.

Take expressions in the form: `derivative(u(x,y), x)` to `derivative(phi, u, [x, y], εs, order, θ)`,
where
 phi - trial solution
 u - function
 x,y - coordinates of point
 εs - epsilon mask
 order - order of derivative
 θ - weight in neural network
"""
function _transform_expression(ex,dict_indvars,dict_depvars, initθ, strategy)
    _args = ex.args
    for (i,e) in enumerate(_args)
        if !(e isa Expr)
            if e in keys(dict_depvars)
                depvar = _args[1]
                num_depvar = dict_depvars[depvar]
                indvars = _args[2:end]
                cord = strategy isa QuadratureTraining ? :([$(indvars...)]) : :cord
                ex.args = if length(dict_depvars) == 1
                    [:u, cord, :($θ), :phi]
                else
                    [:u, cord, Symbol(:($θ),num_depvar), Symbol(:phi,num_depvar)]
                end
                break
            elseif e isa ModelingToolkit.Differential
                derivative_variables = Symbol[]
                order = 0
                while (_args[1] isa ModelingToolkit.Differential)
                    order += 1
                    push!(derivative_variables, toexpr(_args[1].x))
                    _args = _args[2].args
                end
                depvar = _args[1]
                num_depvar = dict_depvars[depvar]
                indvars = _args[2:end]
                cord = strategy  isa QuadratureTraining ? :([$(indvars...)])  : :cord
                dim_l = length(indvars)
                εs = [get_ε(dim_l,d) for d in 1:dim_l]
                undv = [dict_indvars[d_p] for d_p  in derivative_variables]
                εs_dnv = [εs[d] for d in undv]
                ex.args = if length(dict_depvars) == 1
                    [:derivative, :phi, :u, cord, εs_dnv, order, :($θ)]
                else
                    [:derivative, Symbol(:phi,num_depvar), :u, cord, εs_dnv, order, Symbol(:($θ),num_depvar)]
                end
                break
            end
        else
            ex.args[i] = _transform_expression(ex.args[i],dict_indvars,dict_depvars,initθ,strategy)
        end
    end
    return ex
end

"""
Parse ModelingToolkit equation form to the inner representation.

Example:

1)  1-D ODE: Dt(u(t)) ~ t +1

    Take expressions in the form: 'Equation(derivative(u(t), t), t + 1)' to 'derivative(phi, u_d, [t], [[ε]], 1, θ) - (t + 1)'

2)  2-D PDE: Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

    Take expressions in the form:
     Equation(derivative(derivative(u(x, y), x), x) + derivative(derivative(u(x, y), y), y), -(sin(πx)) * sin(πy))
    to
     (derivative(phi,u, [x, y], [[ε,0],[ε,0]], 2, θ) + derivative(phi, u, [x, y], [[0,ε],[0,ε]], 2, θ)) - -(sin(πx)) * sin(πy)

3)  System of PDEs: [Dx(u1(x,y)) + 4*Dy(u2(x,y)) ~ 0,
                    Dx(u2(x,y)) + 9*Dy(u1(x,y)) ~ 0]

    Take expressions in the form:
    2-element Array{Equation,1}:
        Equation(derivative(u1(x, y), x) + 4 * derivative(u2(x, y), y), ModelingToolkit.Constant(0))
        Equation(derivative(u2(x, y), x) + 9 * derivative(u1(x, y), y), ModelingToolkit.Constant(0))
    to
      [(derivative(phi1, u1, [x, y], [[ε,0]], 1, θ1) + 4 * derivative(phi2, u, [x, y], [[0,ε]], 1, θ2)) - 0,
       (derivative(phi2, u2, [x, y], [[ε,0]], 1, θ2) + 9 * derivative(phi1, u, [x, y], [[0,ε]], 1, θ1)) - 0]
"""

function build_symbolic_equation(eq,_indvars,_depvars,initθ,strategy)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    parse_equation(eq,dict_indvars,dict_depvars,initθ,strategy)
end


function parse_equation(eq,dict_indvars,dict_depvars,initθ, strategy)
    eq_lhs = isequal(expand_derivatives(eq.lhs), 0) ? eq.lhs : expand_derivatives(eq.lhs)
    eq_rhs = isequal(expand_derivatives(eq.rhs), 0) ? eq.rhs : expand_derivatives(eq.rhs)

    left_expr = transform_expression(toexpr(eq_lhs),dict_indvars,dict_depvars,initθ,strategy)
    right_expr = transform_expression(toexpr(eq_rhs),dict_indvars,dict_depvars,initθ,strategy)
    loss_func = if strategy isa QuadratureTraining
        :($left_expr - $right_expr)
    else
        left_expr = Broadcast.__dot__(left_expr)
        right_expr = Broadcast.__dot__(right_expr)
        :($left_expr .- $right_expr)
    end
end

"""
Build a loss function for a PDE or a boundary condition

# Examples: System of PDEs:

Take expressions in the form:

[Dx(u1(x,y)) + 4*Dy(u2(x,y)) ~ 0,
 Dx(u2(x,y)) + 9*Dy(u1(x,y)) ~ 0]

to

:((cord, θ, phi, derivative, u)->begin
          #= ... =#
          #= ... =#
          begin
              (θ1, θ2) = (θ[1:33], θ"[34:66])
              (phi1, phi2) = (phi[1], phi[2])
              let (x, y) = (cord[1], cord[2])
                  [(+)(derivative(phi1, u, [x, y], [[ε, 0.0]], 1, θ1), (*)(4, derivative(phi2, u, [x, y], [[0.0, ε]], 1, θ2))) - 0,
                   (+)(derivative(phi2, u, [x, y], [[ε, 0.0]], 1, θ2), (*)(9, derivative(phi1, u, [x, y], [[0.0, ε]], 1, θ1))) - 0]
              end
          end
      end)
"""
function build_symbolic_loss_function(eqs,_indvars,_depvars, phi, derivative,initθ,strategy; bc_indvars=nothing)
    # dictionaries: variable -> unique number
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    bc_indvars = bc_indvars==nothing ? indvars : bc_indvars
    return build_symbolic_loss_function(eqs,indvars,depvars,
                                        dict_indvars,dict_depvars,
                                        phi, derivative,initθ,
                                        strategy;
                                        bc_indvars = bc_indvars)
end

function build_symbolic_loss_function(eqs,indvars,depvars,
                                      dict_indvars,dict_depvars,
                                      phi, derivative, initθ,
                                      strategy;
                                      bc_indvars = indvars)

    vars = :(cord, $θ, phi, derivative,u ,p)
    ex = Expr(:block)
    if length(depvars) != 1
        θ_nums = Symbol[]
        phi_nums = Symbol[]
        for v in depvars
            num = dict_depvars[v]
            push!(θ_nums,:($(Symbol(:($θ),num))))
            push!(phi_nums,:($(Symbol(:phi,num))))
        end

        expr_θ = Expr[]
        expr_phi = Expr[]

        acum =  [0;accumulate(+, length.(initθ))]
        sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]

        for i in eachindex(depvars)
            push!(expr_θ, :($θ[$(sep[i])]))
            push!(expr_phi, :(phi[$i]))
        end

        vars_θ = Expr(:(=), build_expr(:tuple, θ_nums), build_expr(:tuple, expr_θ))
        push!(ex.args,  vars_θ)

        vars_phi = Expr(:(=), build_expr(:tuple, phi_nums), build_expr(:tuple, expr_phi))
        push!(ex.args,  vars_phi)
    end

    if strategy isa QuadratureTraining
        indvars_ex = [:($:cord[$i]) for (i, u) ∈ enumerate(bc_indvars)]
        left_arg_pairs, right_arg_pairs = bc_indvars,indvars_ex
        vars_eq = Expr(:(=), build_expr(:tuple, left_arg_pairs), build_expr(:tuple, right_arg_pairs))
    else
        indvars_ex = [:($:cord[[$i],:]) for (i, u) ∈ enumerate(indvars)]
        left_arg_pairs, right_arg_pairs = indvars,indvars_ex
        vars_eq = Expr(:(=), build_expr(:tuple, left_arg_pairs), build_expr(:tuple, right_arg_pairs))
    end

    let_ex = Expr(:let, vars_eq, loss_function)
    push!(ex.args,  let_ex)

    expr_loss_function = :(($vars) -> begin $ex end)
end

function build_loss_function(eqs,_indvars,_depvars, phi, derivative,initθ;bc_indvars=nothing,params_len=0)
    # dictionaries: variable -> unique number
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    bc_indvars = bc_indvars==nothing ? indvars : bc_indvars
    return build_loss_function(eqs,indvars,depvars,
                               dict_indvars,dict_depvars,
                               phi, derivative,initθ,params_len,
                               bc_indvars = bc_indvars)
end

function build_loss_function(eqs,indvars,depvars,
                             dict_indvars,dict_depvars,
                             phi, derivative, initθ,params_len;
                             bc_indvars = indvars)

     expr_loss_function = build_symbolic_loss_function(eqs,indvars,depvars,
                                                       dict_indvars,dict_depvars,
                                                       phi, derivative, initθ,
                                                       strategy;
                                                       bc_indvars = bc_indvars)
    u = get_u(;strategy=strategy)
    _loss_function = @RuntimeGeneratedFunction(expr_loss_function)
    loss_function = (cord, θ) -> begin
        θ = θ[1:end-params_len]
        p = θ[end-params_len+1:end]
        _loss_function(cord, θ, phi, derivative, u, p)
    end
    return loss_function
end

function get_vars(indvars_, depvars_)
    depvars = [nameof(value(d)) for d in depvars_]
    indvars = [nameof(value(i)) for i in indvars_]
    dict_indvars = get_dict_vars(indvars)
    dict_depvars = get_dict_vars(depvars)
    return depvars,indvars,dict_indvars,dict_depvars
end

function get_varibles(eqs,_indvars::Array,_depvars::Array)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    return get_varibles(eqs,dict_indvars,dict_depvars)
end

function get_varibles(eqs,dict_indvars,dict_depvars)
    bc_args = get_argument(eqs,dict_indvars,dict_depvars)
    return map(barg -> filter(x -> x isa Symbol, barg), bc_args)
end

function get_number(eqs,dict_indvars,dict_depvars)
    bc_args = get_argument(eqs,dict_indvars,dict_depvars)
    return map(barg -> filter(x -> x isa Number, barg), bc_args)
end

function find_thing_in_expr(ex::Expr, thing; ans = Expr[])
    for e in ex.args
        if e isa Expr
            if thing in e.args
                push!(ans,e)
            end
            find_thing_in_expr(e,thing; ans=ans)
        end
    end
    return collect(Set(ans))
end

# Get arguments from boundary condition functions
function get_argument(eqs,dict_indvars,dict_depvars)
    exprs = toexpr.(eqs)
    vars = map(exprs) do expr
        _vars =  map(depvar -> find_thing_in_expr(expr,  depvar), collect(keys(dict_depvars)))
        f_vars = filter(x -> !isempty(x), _vars)
        map(x -> first(x), f_vars)
    end
    args_ = map(vars) do _vars
        map(var -> var.args[2:end] , _vars)
    end
    return first.(args_) #TODO for all arguments
end

function generate_training_sets(domains,dx,eqs,bcs,_indvars::Array,_depvars::Array)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    return generate_training_sets(domains,dx,eqs,bcs,dict_indvars,dict_depvars)
end
# Generate training set in the domain and on the boundary
function generate_training_sets(domains,dx,eqs,bcs,dict_indvars::Dict,dict_depvars::Dict)
    if dx isa Array
        dxs = dx
    else
        dxs = fill(dx,length(domains))
    end

    spans = [d.domain.lower:dx:d.domain.upper for (d,dx) in zip(domains,dxs)]
    dict_var_span = Dict([Symbol(d.variables) => d.domain.lower:dx:d.domain.upper for (d,dx) in zip(domains,dxs)])

    bound_args = get_argument(bcs,dict_indvars,dict_depvars)
    bound_vars = get_varibles(bcs,dict_indvars,dict_depvars)

    dif = [Float32[] for i=1:size(domains)[1]]
    for _args in bound_args
        for (i,x) in enumerate(_args)
            if x isa Number
                push!(dif[i],x)
            end
        end
    end
    cord_train_set = collect.(spans)
    bc_data = map(zip(dif,cord_train_set)) do (d,c)
        setdiff(c, d)
    end

    dict_var_span_ = Dict([Symbol(d.variables) => bc for (d,bc) in zip(domains,bc_data)])

    bcs_train_sets = map(bound_args) do bt
        span = map(b -> get(dict_var_span, b, b), bt)
        _set = Float32.(hcat(vec(map(points -> collect(points), Iterators.product(span...)))...))
    end

    pde_vars = get_varibles(eqs,dict_indvars,dict_depvars)
    pde_args = get_argument(eqs,dict_indvars,dict_depvars)

    pde_train_set = Float32.(hcat(vec(map(points -> collect(points), Iterators.product(bc_data...)))...))

    pde_train_sets = map(pde_args) do bt
        span = map(b -> get(dict_var_span_, b, b), bt)
        _set = Float32.(hcat(vec(map(points -> collect(points), Iterators.product(span...)))...))
    end
    [pde_train_sets,bcs_train_sets]
end

function get_bounds(domains,eqs,bcs,_indvars::Array,_depvars::Array)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    return get_bounds(domains,eqs,bcs,dict_indvars,dict_depvars)
end

function get_bounds(domains,bcs,_indvars::Array,_depvars::Array,strategy::QuadratureTraining)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(_indvars, _depvars)
    return get_bounds(domains,bcs,dict_indvars,dict_depvars,strategy)
end

function get_bounds(domains,bcs,dict_indvars,dict_depvars,strategy::QuadratureTraining)
    bound_vars = get_varibles(bcs,dict_indvars,dict_depvars)

    pde_lower_bounds = [d.domain.lower for d in domains]
    pde_upper_bounds = [d.domain.upper for d in domains]
    pde_bounds= [[pde_lower_bounds],[pde_upper_bounds]]

    dict_lower_bound = Dict([Symbol(d.variables) => d.domain.lower for d in domains])
    dict_upper_bound = Dict([Symbol(d.variables) => d.domain.upper for d in domains])

    bcs_lower_bounds = map(bound_vars) do bt
        map(b -> dict_lower_bound[b], bt)
    end
    bcs_upper_bounds = map(bound_vars) do bt
        map(b -> dict_upper_bound[b], bt)
    end
    bcs_bounds= [bcs_lower_bounds,bcs_upper_bounds]

    [pde_bounds, bcs_bounds]
end

function get_bounds(domains,eqs,bcs,dict_indvars,dict_depvars)
    dict_span = Dict([Symbol(d.variables) => [d.domain.lower, d.domain.upper] for d in domains])
    # pde_bounds = [[d.domain.lower,d.domain.upper] for d in domains]
    pde_args = get_argument(eqs,dict_indvars,dict_depvars)

    pde_bounds= map(pde_args) do pd
        span = map(p -> get(dict_span, p, p), pd)
    end

    bound_args = get_argument(bcs,dict_indvars,dict_depvars)
    dict_span = Dict([Symbol(d.variables) => [d.domain.lower, d.domain.upper] for d in domains])

    bcs_bounds= map(bound_args) do bt
        span = map(b -> get(dict_span, b, b), bt)
    end
    [pde_bounds,bcs_bounds]
end

function get_phi(chain)
    # The phi trial solution
    if chain isa FastChain
        phi = (x,θ) -> chain(adapt(typeof(θ),x),θ)
    else
        _,re  = Flux.destructure(chain)
        phi = (x,θ) -> re(θ)(adapt(typeof(θ),x))
    end
    phi
end

function get_u(;strategy=nothing)
    if strategy isa QuadratureTraining
	    u = (cord, θ, phi)-> sum(phi(cord, θ))
    else
        u = (cord, θ, phi)-> phi(cord, θ)
    end
end

Base.Broadcast.broadcasted(::typeof(get_u()), cord, θ, phi) = get_u()(cord, θ, phi)

# the method to calculate the derivative
function get_numeric_derivative(;vectorize = true)
    _epsilon = 1 / (2*cbrt(eps(Float32)))
    derivative = if vectorize == true
        (phi,u,x,εs,order,θ) ->
        begin
            ε = εs[order]
            ε = adapt(typeof(θ),ε)
            x = adapt(typeof(θ),x)
            if order > 1
                return (derivative(phi,u,x .+ ε,εs,order-1,θ)
                      .- derivative(phi,u,x .- ε,εs,order-1,θ)) .* _epsilon
            else
                return (u(x .+ ε,θ,phi) .- u(x .- ε,θ,phi)) .* _epsilon
            end
        end
    else
        (phi,u,x,εs,order,θ) ->
        begin
            ε = εs[order]
            ε = adapt(typeof(θ),ε)
            x = adapt(typeof(θ),x)
            if order > 1
                return (derivative(phi,u,x+ε,εs,order-1,θ)
                      - derivative(phi,u,x-ε,εs,order-1,θ)) * _epsilon
            else
                return (u(x+ε,θ,phi) - u(x-ε,θ,phi)) * _epsilon
            end
        end
    end
    derivative
end

Base.Broadcast.broadcasted(::typeof(get_numeric_derivative()), phi,u,x,εs,order,θ) = get_numeric_derivative()(phi,u,x,εs,order,θ)

function get_loss_function(loss_functions, train_sets, strategy::GridTraining;τ=nothing)
    if τ == nothing
        τs_ = [size(set)[2] for set in train_sets]
        τs = 1.0f0 ./ τs_
    end

    function inner_loss(loss_function,x,θ)
        sum(abs2,loss_function(x, θ))
    end

    if !(loss_functions isa Array)
        loss_functions = [loss_functions]
        train_sets = [train_sets]
    end
    f = (loss,train_set,θ) -> sum([inner_loss(loss,x,θ) for x in train_set])
    loss = (θ) ->  τ * sum(f(loss_function,train_set,θ) for (loss_function,train_set) in zip(loss_functions,train_sets))
    return loss
end

function generate_random_points(points, bound)
    function f(b)
      if b isa Number
           fill(Float32(b),(1,points))
       else
           lb, ub =  b[1], b[2]
           lb .+ ub .* Float32.(rand(1,points))
       end
    end
    vcat(f.(bound)...)
end

function get_loss_function(loss_functions, bounds, strategy::StochasticTraining;τ=nothing)
    points = strategy.points

    function inner_loss(loss_function,x,θ)
        sum(abs2,loss_function(x, θ))
    end

    if !(loss_functions isa Array)
        loss_functions = [loss_functions]
        lbs = [lbs]
        ubs = [ubs]
    end
    if τ == nothing
        τ = 1.0f0 / points
    end

    loss = (θ) -> begin
        total = 0.
        for (lb, ub,l) in zip(lbs, ubs, loss_functions)
            len = length(lb)
            for i in 1:points
                r_point = lb .+ ub .* rand(len)
                total += inner_loss(l,r_point,θ)
            end
        end
        return total
    end

    return loss
end

function get_loss_function(loss_functions, bounds, strategy::QuasiRandomTraining;τ=nothing)
    sampling_alg = strategy.sampling_alg
    points = strategy.points
    minibatch = strategy.minibatch

    function inner_loss(loss_function,x,θ)
        sum(abs2,loss_function(x, θ))
    end

    if !(loss_functions isa Array)
        loss_functions = [loss_functions]
        lbs = [lbs]
        ubs = [ubs]
    end
    if τ == nothing
        τ = 1.0f0 / points
    end
    ss =[]
    for (lb, ub) in zip(lbs, ubs)
        s = QuasiMonteCarlo.generate_design_matrices(points,lb,ub,sampling_alg,minibatch)
        push!(ss,s)
    end
    loss = (θ) -> begin
        total = 0.
        for (lb, ub,s_,l) in zip(lbs,ubs,ss,loss_functions)
            s =  s_[rand(1:minibatch)]
            step_ = size(lb)[1]
            k = size(lb)[1]-1
            for i in 1:step_:step_*points
                r_point = s[i:i+k]
                total += inner_loss(l,r_point,θ)
            end
        end
        return total
    end
    return loss
end


function get_loss_function(loss_functions, bounds, strategy::QuadratureTraining;τ=nothing)
    lbs,ubs = bounds
    if τ == nothing
        τ = 1.0f0
    end

    function inner_loss(loss_function,x,θ)
        sum(abs2,loss_function(x, θ))
    end

    if !(loss_functions isa Array)
        loss_functions = [loss_functions]
        lbs = [lbs]
        ubs = [ubs]
    end

    f = (lb,ub,loss_,θ) -> begin
        _loss = (x,θ) -> inner_loss(loss_, x, θ)
        prob = QuadratureProblem(_loss,lb,ub,θ;batch = strategy.batch)
        abs(solve(prob,
              strategy.quadrature_alg,
              reltol = strategy.reltol,
              abstol = strategy.abstol,
              maxiters = strategy.maxiters)[1])
    end
    loss = (θ) -> τ*sum(f(lb,ub,loss_,θ) for (lb,ub,loss_) in zip(lbs,ubs,loss_functions))
    return loss
end
function symbolic_discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
    eqs = pde_system.eqs
    bcs = pde_system.bcs

    domains = pde_system.domain
    # dimensionality of equation
    dim = length(domains)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(pde_system.indvars, pde_system.depvars)

    chain = discretization.chain
    initθ = discretization.init_params
    phi = discretization.phi
    derivative = discretization.derivative
    strategy = discretization.strategy
    if !(eqs isa Array)
        eqs = [eqs]
    end
    symbolic_pde_loss_functions = [build_symbolic_loss_function(eq,indvars,depvars,
                                                              dict_indvars,dict_depvars,
                                                              phi, derivative,initθ,strategy) for eq  in eqs]

    bc_indvars = get_varibles(bcs,dict_indvars,dict_depvars)
    symbolic_bc_loss_functions = [build_symbolic_loss_function(bc,indvars,depvars,
                                                               dict_indvars,dict_depvars,
                                                               phi, derivative,initθ,strategy;
                                                               bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]
    symbolic_pde_loss_functions,symbolic_bc_loss_functions
end

# Convert a PDE problem into an OptimizationProblem
function DiffEqBase.discretize(pde_system::PDESystem, discretization::PhysicsInformedNN)
    eqs = pde_system.eqs
    bcs = pde_system.bcs

    domains = pde_system.domain
    p = pde_system.ps
    params_len =  if p isa SciMLBase.NullParameters 0 else length(p) end
    # dimensionality of equation
    dim = length(domains)
    depvars,indvars,dict_indvars,dict_depvars = get_vars(pde_system.indvars,pde_system.depvars)

    chain = discretization.chain
    initθ = discretization.init_params
    flat_initθ = if length(depvars) != 1 vcat(initθ...) else  initθ end
    flat_initθ = if p isa SciMLBase.NullParameters flat_initθ else vcat(flat_initθ , p) end

    phi = discretization.phi
    derivative = discretization.derivative
    strategy = discretization.strategy
    if !(eqs isa Array)
        eqs = [eqs]
    end
    _pde_loss_functions = [build_loss_function(eq,indvars,depvars,
                                             dict_indvars,dict_depvars,
                                             phi, derivative, initθ,strategy) for eq in eqs]

    _pde_loss_function = build_loss_function(eqs,indvars,depvars,
                                                 dict_indvars,dict_depvars,phi, derivative, initθ, params_len)

    bc_indvars = get_bc_varibles(bcs,dict_indvars,dict_depvars)
    _bc_loss_functions = [build_loss_function(bc,indvars,depvars,
                                                  dict_indvars,dict_depvars,
                                                  phi, derivative, initθ, params_len;
                                                  bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]

    pde_loss_function, bc_loss_function =
    if strategy isa GridTraining
        dx = strategy.dx

        train_sets = generate_training_sets(domains,dx,eqs,bcs,
                                            dict_indvars,dict_depvars)

        # the points in the domain and on the boundary
        pde_train_sets, bcs_train_sets = train_sets

        pde_train_sets = adapt.(typeof(flat_initθ),pde_train_sets)
        bcs_train_sets =  adapt.(typeof(flat_initθ),bcs_train_sets)

        pde_loss_function = get_loss_function(_pde_loss_functions,
                                                        pde_train_sets,
                                                        strategy)

        bc_loss_function = get_loss_function(_bc_loss_functions,
                                                       bcs_train_sets,
                                                       strategy)
        (pde_loss_function, bc_loss_function)
    elseif strategy isa StochasticTraining
          bounds = get_bounds(domains,eqs,bcs,dict_indvars,dict_depvars)
          pde_bounds, bcs_bounds = bounds
          pde_loss_function = get_loss_function(_pde_loss_functions,
                                                          pde_bounds,
                                                          strategy)

          pde_dim = size(pde_bounds[1])[1] #TODO
          bcs_dim = isempty(maximum(size.(bcs_bounds[1]))) ? nothing : maximum(size.(bcs_bounds))[1]
          bcs_cond_size = size(bcs_bounds)[1]

          points = bcs_dim == nothing ? 1 : bcs_cond_size*Int(round(strategy.points^(bcs_dim/pde_dim)))
          strategy_ = StochasticTraining(points)

          bc_loss_function = get_loss_function(_bc_loss_functions,
                                                         bcs_bounds,
                                                         strategy_)
          (pde_loss_function, bc_loss_function)
    elseif strategy isa QuasiRandomTraining
         bounds = get_bounds(domains,eqs,bcs,dict_indvars,dict_depvars)
         pde_bounds, bcs_bounds = bounds
         pde_loss_function = get_loss_function(_pde_loss_functions,
                                                        pde_bounds,
                                                        strategy)

         pde_dim = size(pde_bounds[1])[1] #TODO
         bcs_dim = isempty(maximum(size.(bcs_bounds[1]))) ? nothing : maximum(size.(bcs_bounds))[1]
         bcs_cond_size = size(bcs_bounds)[1]

         points = bcs_dim == nothing ? 1 : bcs_cond_size*Int(round(strategy.points^(bcs_dim/pde_dim)))
         strategy_ = QuasiRandomTraining(points;
                                        sampling_alg = strategy.sampling_alg,
                                        minibatch = strategy.minibatch)

         bc_loss_function = get_loss_function(_bc_loss_functions,
                                                       bcs_bounds,
                                                       strategy_)
         (pde_loss_function, bc_loss_function)
    elseif strategy isa QuadratureTraining
        bounds = get_bounds(domains,bcs,dict_indvars,dict_depvars,strategy)
        pde_bounds, bcs_bounds = bounds
        plbs,pubs = pde_bounds
        blbs,bubs = bcs_bounds
        pl = length(plbs[1])
        bl = length(blbs[1])
        bsl = length(blbs)

        τ_ = (10)^pl
        τp = 1.0f0 / τ_

        pde_loss_function = get_loss_function(_pde_loss_functions,
                                              pde_bounds,
                                              strategy;
                                              τ=τp)

        τb =  1.0f0 / (bsl * τ_^(bl/pl))

        if bl == 0
            _bc_loss_functions = [build_loss_function(bc,indvars,depvars,
                                                          dict_indvars,dict_depvars,
                                                          phi, get_numeric_derivative(;vectorize = true), initθ, GridTraining(0.1);
                                                          bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]
            train_sets = generate_training_sets(domains,0.1,eqs,bcs,
                                                dict_indvars,dict_depvars)
            pde_train_set, bcs_train_sets = train_sets
            bc_loss_function = get_loss_function(_bc_loss_functions,
                                                 bcs_train_sets,
                                                 GridTraining(0.1))

        elseif bl == 1 && strategy.quadrature_alg in [CubaCuhre(),CubaDivonne()]
            @warn "$(strategy.quadrature_alg) does not work with one-dimensional
            problems, so for the boundary conditions loss function,
            the quadrature algorithm was replaced by HCubatureJL"

            strategy = QuadratureTraining(quadrature_alg = HCubatureJL(),
                                          reltol = bsl*(strategy.reltol)^(bl/pl),
                                          abstol = bsl*(strategy.abstol)^(bl/pl),
                                          maxiters = bsl*Int(round((strategy.maxiters)^(bl/pl))),
                                          batch = strategy.batch)
            bc_loss_function = get_loss_function(_bc_loss_functions,
                                                 bcs_bounds,
                                                 strategy;
                                                 τ=τb)
        else
            strategy = QuadratureTraining(quadrature_alg = strategy.quadrature_alg,
                                          reltol = bsl*(strategy.reltol)^(bl/pl),
                                          abstol = bsl*(strategy.abstol)^(bl/pl),
                                          maxiters = bsl*Int(round((strategy.maxiters)^(bl/pl))),
                                          batch = strategy.batch)
            bc_loss_function = get_loss_function(_bc_loss_functions,
                                             bcs_bounds,
                                             strategy;
                                             τ=τb)
        end
        (pde_loss_function, bc_loss_function)
    end

    function loss_function_(θ,p)
        return pde_loss_function(θ) + bc_loss_function(θ)
    end

    f = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
    GalacticOptim.OptimizationProblem(f, flat_initθ)
end
