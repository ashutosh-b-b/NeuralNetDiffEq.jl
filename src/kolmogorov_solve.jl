struct NNKolmogorov{C,O} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    
end
NNKolmogorov(chain ; opt=Flux.ADAM(0.1)) = NNKolmogorov(chain , opt)

function DiffEqBase.solve(
    prob::KolmogorovPDEProblem,
    alg::NNKolmogorov,
    dt,
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100,
    args...;
    )


    tspan = prob.tspan
    phi = prob.phi
    xspan = prob.xspan
    T = tspan[2]

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    ps     = Flux.params(chain)
    xi     = randn(Uniform(xspan[0] , xspan[1]))
    N = Normal(0 , sqrt(2. *T ))
    x_sde = pdf.(N , xi)
    y = phi(x_sde)
    data   = Iterators.repeated((xi , y), maxiters)
    
    #MSE Loss Function
    loss(x , y) =Flux.mse(chain(xi), y)

    cb = function ()
        l = loss()
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end
        Flux.train!(loss, ps, data, opt; cb = cb)
        
        chain(x_sde)

   
end #solve
