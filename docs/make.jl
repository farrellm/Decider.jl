using Decider
using Documenter

DocMeta.setdocmeta!(Decider, :DocTestSetup, :(using Decider); recursive = true)

# The home page is the README, generated here so there is no second copy to
# drift. `docs/src/index.md` is generated and gitignored; edit README.md.
#
# The title gains the .jl and the badges (GitHub furniture) go. GitHub renders
# a `jldoctest` block as plain text, so the README writes examples as `julia`
# blocks with a `# output` section, and they become one doctest session here,
# sharing their variables in order.
function readme_as_index(readme, index)
    text = read(readme, String)
    text = replace(text, r"^# Decider\n" => "# Decider.jl\n")
    text = replace(text, r"^\[!\[[^\n]*\n"m => "")
    text = replace(text, r"\n{3,}" => "\n\n")   # the blank run the badges left
    text = replace(text,
        r"```julia\n((?:(?!```)[\s\S])*?\n# output\n[\s\S]*?)```" =>
            s"```jldoctest readme\n\1```")
    return write(index, text)
end

readme_as_index(joinpath(@__DIR__, "..", "README.md"),
    joinpath(@__DIR__, "src", "index.md"))

makedocs(;
    modules = [Decider],
    authors = "Matthew Farrell",
    sitename = "Decider.jl",
    format = Documenter.HTML(;
        canonical = "https://farrellm.github.io/Decider.jl",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/farrellm/Decider.jl",
    devbranch = "master",
)
