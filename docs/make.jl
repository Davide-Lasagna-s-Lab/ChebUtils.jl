using Documenter, ChebUtils, LinearAlgebra

makedocs(
    sitename = "ChebUtils.jl",
    modules  = [ChebUtils],
    authors  = "Davide Lasagna",
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://Davide-Lasagna-s-Lab.github.io/ChebUtils.jl/stable",
    ),
    pages = [
        "Home"                   => "index.md",
        "Concepts & Conventions" => "guide.md",
        "API Reference"          => "api.md",
    ],
    checkdocs = :exports,
    warnonly  = false,
)

deploydocs(
    repo   = "github.com/Davide-Lasagna-s-Lab/ChebUtils.jl.git",
    target = "build",
)
