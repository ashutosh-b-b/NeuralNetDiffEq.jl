struct NNKolmogorov{C,O,S,E} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    sdealg::S
    ensemblealg::E
end
NNKolmogorov(chain  ; opt=Flux.ADAM(0.1) , sdealg = EM() , ensemblealg = EnsembleThreads()) = NNKolmogorov(chain , opt , sdealg , ensemblealg)

function DiffEqBase.solve(
    prob::Union{KolmogorovPDEProblem,SDEProblem},
    alg::NNKolmogorov;
    abstol = 1f-6,
    verbose = false,
    maxiters = 300,
    trajectories = 1000,
    save_everystep = false,
    dt,
    dx,
    check = false,
    lambda,
    kwargs...
    )

    tspan = prob.tspan
    sigma = prob.g
    μ = prob.f
    noise_rate_prototype = prob.noise_rate_prototype
    if prob isa SDEProblem
        xspan = prob.kwargs.data.xspan
        d = prob.kwargs.data.d
        u0 = prob.u0
        phi(xi) = pdf(u0 , xi)
    else
        xspan = prob.xspan
        d = prob.d
        phi = prob.phi
    end
    ts = tspan[1]:dt:tspan[2]
    xs = xspan[1]:dx:xspan[2]
    N = size(ts)
    T = tspan[2]

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    sdealg = alg.sdealg
    ensemblealg = alg.ensemblealg
    ps     = Flux.params(chain)
    xi     = rand(xs ,d ,trajectories)
    #Finding Solution to the SDE having initial condition xi. Y = Phi(S(X , T))
    sdeproblem = SDEProblem(μ,sigma,xi,tspan,noise_rate_prototype = noise_rate_prototype)
    function prob_func(prob,i,repeat)
      SDEProblem(prob.f , prob.g , xi[: , i] , prob.tspan ,noise_rate_prototype = prob.noise_rate_prototype)
    end
    output_func(sol,i) = (sol[end],false)
    ensembleprob = EnsembleProblem(sdeproblem , prob_func = prob_func , output_func = output_func)
    sim = solve(ensembleprob, sdealg, ensemblealg , dt=dt, trajectories=trajectories,adaptive=false)
    # sol = solve(sdeproblem, sdealg ,dt=0.01 , save_everystep=false , kwargs...)
    # x_sde = sol[end]
    # for u in sim.u
    #     x_sde = hcat(x_sde , u)
    # end
    # x_sde = fill(0.00 , 1 , length(sim.u))
    # for i  in (1:length(sim.u))
    #     for j in (1:length(sim.u[i]))
    #         x_sde[j , i] = sim.u[i][j]
    #     end
    # end
    x_sde = reduce(hcat , sim.u)
    y = phi(x_sde)
    println(size(y))
    # if check == true
    #     yi = reshape(y , size(y)[2])
    #     xi = reshape(xi , size(xi)[2])
    #     _zscore = zscore(yi) .< 6
    #     xi = xi[_zscore]
    #     yi = yi[_zscore]
    #     _zscore = zscore(yi) .> -6
    #     xi = xi[_zscore]
    #     yi = yi[_zscore]
    #     yi = reshape(yi , d , length(yi))
    #     xi = reshape(xi , d , length(xi))
    # end
    # println(size(yi))


    data   = Iterators.repeated((xi , y), maxiters)

    #MSE Loss Function
    L1(x) = sum(abs.(x))
    loss(x , y) =Flux.mse(chain(x), y) + lambda*sum(L1  , Flux.params(chain))
    # loss(x , y) = sum( (abs(xn - yn))^1.5 for (xn , yn) in (xi, y))

    cb = function ()
        l = loss(xi, y)
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end

    Flux.train!(loss, ps, data, opt; cb = cb)
    xi , chain(xi), y , x_sde
 end #solve
