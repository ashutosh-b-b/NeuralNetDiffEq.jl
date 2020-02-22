module NeuralNetDiffEq

using Reexport, Statistics
@reexport using DiffEqBase
using Flux, Zygote, DiffEqSensitivity, Distributions
import Tracker

abstract type NeuralNetDiffEqAlgorithm <: DiffEqBase.AbstractODEAlgorithm end

struct TerminalPDEProblem{G,F,Mu,Sigma,X,T,P} <: DiffEqBase.DEProblem
    g::G
    f::F
    μ::Mu
    σ::Sigma
    X0::X
    tspan::Tuple{T,T}
    p::P
    TerminalPDEProblem(g,f,μ,σ,X0,tspan,p=nothing) = new{typeof(g),typeof(f),
                                                         typeof(μ),typeof(σ),
                                                         typeof(X0),eltype(tspan),
                                                         typeof(p)}(
                                                         g,f,μ,σ,X0,tspan,p)
end

Base.summary(prob::TerminalPDEProblem) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::TerminalPDEProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
end

struct KolmogorovPDEProblem{Phi , X , T } <: DiffEqBase.DEProblem
    phi::Phi
    xspan::Tuple{X,X}
    tspan::Tuple{T,T}
    KolmogorovPDEProblem(phi , xspan , tspan) = new{typeof(phi),eltype(tspan),eltype(xspan)}(
                                                         phi , xspan , tspan)
end

Base.summary(prob::KolmogorovPDEProblem) = string(nameof(typeof(prob)))
function Base.show(io::IO, A::KolmogorovPDEProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
  print(io,"xspan: ")
  show(io,A.xspan)
end

include("ode_solve.jl")
include("pde_solve.jl")
include("pde_solve_ns.jl")
include("rode_solve.jl")
include("kolmogorov_solve.jl")


export NNODE, TerminalPDEProblem, NNPDEHan, NNPDENS, NNRODE, KolmogorovPDEProblem, NNKolmogorov

end # module
