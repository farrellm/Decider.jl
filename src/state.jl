"""
    Image(bytes::Vector{UInt8}, mime)
    Image(path::AbstractString)

An image to include in the state, as an element of a `Vector` state:
`[\"What is in this picture?\", Image(\"cat.png\")]`. Only backends that accept images
(currently [`OpenAIDecisions`](@ref)) can send them; others throw `ArgumentError` before
making a request. The path form infers the MIME type from the extension (png, jpg/jpeg,
gif, webp).
"""
struct Image
    data::Vector{UInt8}
    mime::String
end

Image(data::AbstractVector{UInt8}, mime::MIME) = Image(Vector{UInt8}(data), string(mime))
Image(data::AbstractVector{UInt8}, mime::AbstractString) =
    Image(Vector{UInt8}(data), String(mime))

const IMAGE_MIMES = Dict(
    ".png" => "image/png",
    ".jpg" => "image/jpeg",
    ".jpeg" => "image/jpeg",
    ".gif" => "image/gif",
    ".webp" => "image/webp",
)

function Image(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    mime = get(IMAGE_MIMES, ext, nothing)
    mime === nothing && throw(
        ArgumentError(
            "cannot infer an image MIME type from \"$path\"; use Image(bytes, mime)"),
    )
    return Image(read(path), mime)
end

data_url(img::Image) = "data:$(img.mime);base64,$(base64encode(img.data))"

Base.show(io::IO, img::Image) =
    print(io, "Image(", length(img.data), " bytes, ", img.mime, ")")

# Images may only appear as elements of a top-level Vector state.
_contains_image(::Image) = true
_contains_image(x::Union{AbstractVector,Tuple}) = any(_contains_image, x)
_contains_image(x::AbstractDict) = any(_contains_image, values(x))
_contains_image(x::NamedTuple) = any(_contains_image, values(x))
_contains_image(::Any) = false

function check_state(backend, state)
    _contains_image(state) || return nothing
    supports_images(backend) || throw(
        ArgumentError(
            "$(nameof(typeof(backend))) does not accept images in the state"),
    )
    if !(state isa AbstractVector) || any(x -> !(x isa Image) && _contains_image(x), state)
        throw(ArgumentError("Image values may only appear as elements of a Vector state"))
    end
    return nothing
end
