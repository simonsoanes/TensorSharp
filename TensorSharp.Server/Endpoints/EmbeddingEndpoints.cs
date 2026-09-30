using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using TensorSharp.Server.ProtocolAdapters;

namespace TensorSharp.Server.Endpoints;

/// <summary>Embedding inference and model discovery for a standalone ASP.NET Core encoder host.</summary>
public static class EmbeddingEndpoints
{
    /// <summary>
    /// Map an embedding-only API after AddTensorSharpEmbeddings. This includes the embedding
    /// and discovery routes in MapOpenAIEndpoints/MapOllamaEndpoints; use this extension
    /// instead of those extensions when the application has no chat services.
    /// </summary>
    public static IEndpointRouteBuilder MapEmbeddingEndpoints(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapPost("/v1/embeddings", (HttpContext context, EmbeddingAdapter adapter) => adapter.OpenAIAsync(context));
        endpoints.MapPost("/api/embed", (HttpContext context, EmbeddingAdapter adapter) => adapter.OllamaAsync(context));
        endpoints.MapGet("/v1/models", (EmbeddingAdapter adapter) => adapter.ListModels());
        endpoints.MapGet("/api/tags", (EmbeddingAdapter adapter) => adapter.GetTags());
        endpoints.MapGet("/api/models", (EmbeddingAdapter adapter) => adapter.GetWebModels());
        endpoints.MapPost("/api/show", (HttpContext context, EmbeddingAdapter adapter) => adapter.ShowAsync(context));
        endpoints.MapGet("/api/version", () => Results.Json(new { version = "0.1.0" }));
        return endpoints;
    }
}
