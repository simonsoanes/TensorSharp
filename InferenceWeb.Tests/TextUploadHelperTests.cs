namespace InferenceWeb.Tests;

public class TextUploadHelperTests
{
    [Fact]
    public void PreserveFullText_DoesNotApplyCharacterOrTokenLimit()
    {
        string content = new('x', 200_000);

        string result = TextUploadHelper.PreserveFullText(content);

        Assert.Same(content, result);
        Assert.Equal(200_000, result.Length);
    }

    [Fact]
    public void PreserveFullText_NormalizesNullToEmptyString()
    {
        string result = TextUploadHelper.PreserveFullText(null!);

        Assert.Same(string.Empty, result);
    }
}
