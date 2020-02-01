struct NNODE{C,O} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
end
NNODE(chain;opt=Flux.ADAM(0.1)) = NNODE(chain,opt)

function DiffEqBase.solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::NeuralNetDiffEqAlgorithm,
    args...;
    dt,
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100)

    DiffEqBase.isinplace(prob) && error("Only out-of-place methods are allowed!")

    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t0 = tspan[1]

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    ps     = Flux.params(chain)
    data   = Iterators.repeated((), maxiters)

    #train points generation
    ts = tspan[1]:dt:tspan[2]

    #The phi trial solution
    if u0 isa Number
        phi = t -> u0 + (t - tspan[1])*first(chain([t]))
    else
        phi = t -> u0 + (t - tspan[1])*chain([t])
    end

    dfdx(t) = ForwardDiff.derivative(phi,t)
    loss() = sum(abs2,sum(abs2,dfdx(t) - f(phi(t),p,t)) for t in ts)

    cb = function ()
        l = loss()
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end
    Flux.train!(loss, ps, data, opt; cb = cb)

    #solutions at timepoints
    if u0 isa Number
        u = [first(phi(t)) for t in ts]
    else
        u = [phi(t) for t in ts]
    end

    sol = DiffEqBase.build_solution(prob,alg,ts,u,calculate_error = false)
    DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    sol
end #solve
