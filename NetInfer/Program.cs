
using NetInfer.Tokenization;
using System.Text;
using Terminal.Gui.App;
using Terminal.Gui.Drivers;
using Terminal.Gui.Editor;
using Terminal.Gui.ViewBase;
using Terminal.Gui.Views;

namespace NetInfer;

internal static class Program
{
#if false
    private static void Main()
    {
        var model = GgufFile.Load(@"D:\Dev\models\Llama-TinyStories.f16.gguf");
        var metadata = model.Metadata.ToDictionary(m => m.Key, m => m);

        var tokens = (GgufMetadataArray<string>)metadata["tokenizer.ggml.tokens"];
        var types = (GgufMetadataArray<int>)metadata["tokenizer.ggml.token_type"];
        var merges = (GgufMetadataArray<string>)metadata["tokenizer.ggml.merges"];
        var unknownTokenId = ((GgufMetadataUint32)metadata["tokenizer.ggml.unknown_token_id"]).Value;

        var tokenized = Tokenizer.Tokenize("Hello, the dog is cute", tokens.Values, types.Values, merges.Values, unknownTokenId);
    }
#else
    private static TextField? _pathField;

    private static void Main()
    {
        using IApplication app = Application.Create();
        app.Init();
        app.Paste += (_, paste) =>
        {
            if (_pathField is not { } pathField)
                return;

            paste.Handled = true;
            InsertPastedPath(pathField, paste.Text);
        };

        using var window = new Window
        {
            Title = "NetInfer - GGUF Explorer",
            Width = Dim.Fill(),
            Height = Dim.Fill(),
        };

        string lastPath = LoadLastPath(out string? configError);
        ShowFileSelector(app, window, lastPath, configError);
        app.Run(window);
    }

    private static void ShowFileSelector(IApplication app, Window window, string initialPath = "", string? error = null)
    {
        window.RemoveAll();

        var page = new View
        {
            CanFocus = true,
            Width = Dim.Fill(),
            Height = Dim.Fill(),
        };

        page.Add(new Label
        {
            Text = "Enter the path to a GGUF file and press Enter:",
            X = 2,
            Y = 1,
            Width = Dim.Fill(2),
        });

        var pathField = new TextField
        {
            Value = initialPath,
            X = 2,
            Y = 3,
            Width = Dim.Fill(2),
        };
        page.Add(pathField);

        var status = new Label
        {
            Text = error ?? string.Empty,
            X = 2,
            Y = 5,
            Width = Dim.Fill(2),
            Height = Dim.Auto(),
        };
        page.Add(status);

        page.Add(new Label
        {
            Text = "Press Esc to exit.",
            X = 2,
            Y = Pos.AnchorEnd(1),
            Width = Dim.Fill(2),
        });

        pathField.KeyDown += (_, key) =>
        {
            if (key.IsCtrl && key.KeyCode == KeyCode.V)
            {
                key.Handled = true;
                if (app.Clipboard is { } clipboard &&
                    clipboard.TryGetClipboardData(out string clipboardText))
                {
                    InsertPastedPath(pathField, clipboardText);
                    status.Text = string.Empty;
                }
                else
                {
                    status.Text = "Could not read text from the system clipboard.";
                }
            }
            else if (key.KeyCode == KeyCode.Esc)
            {
                key.Handled = true;
                app.RequestStop();
            }
            else if (key.KeyCode == KeyCode.Enter)
            {
                key.Handled = true;
                string path = pathField.Value ?? string.Empty;
                GgufFile? file = TryLoadGguf(path, out string? loadError);
                if (file is null)
                {
                    ShowFileSelector(app, window, path, loadError);
                    return;
                }

                string? saveError = SaveLastPath(path.Trim('"'));
                ShowExplorerMenu(app, window, file, path, saveError);
            }
        };

        window.Add(page);
        _pathField = pathField;
        pathField.SetFocus();
    }

    private static void ShowExplorerMenu(IApplication app, Window window, GgufFile file, string path, string? notice = null)
    {
        _pathField = null;
        window.RemoveAll();

        var page = new View
        {
            CanFocus = true,
            Width = Dim.Fill(),
            Height = Dim.Fill(),
        };
        page.Add(new Label
        {
            Text = $"Loaded: {path}",
            X = 2,
            Y = 1,
            Width = Dim.Fill(2),
        });

        if (!string.IsNullOrEmpty(notice))
        {
            page.Add(new Label
            {
                Text = notice,
                X = 2,
                Y = 3,
                Width = Dim.Fill(2),
            });
        }

        var choices = new ListView();
        choices.SetSource(["Metadata exploration", "Tensor exploration"]);
        choices.SelectedItem = 0;
        choices.X = 2;
        choices.Y = string.IsNullOrEmpty(notice) ? 3 : 5;
        choices.Width = Dim.Fill(2);
        choices.Height = Dim.Fill(2);
        choices.KeyDown += (_, key) =>
        {
            if (key.KeyCode == KeyCode.Esc)
            {
                key.Handled = true;
                ShowFileSelector(app, window, path);
            }
            else if (key.KeyCode == KeyCode.Enter)
            {
                key.Handled = true;
                if (choices.SelectedItem == 0)
                    ShowMetadata(app, window, file, path);
                else if (choices.SelectedItem == 1)
                    ShowTensors(app, window, file, path);
            }
        };
        page.Add(choices);

        page.Add(new Label
        {
            Text = "Use Up/Down and Enter to select. Press Esc to choose another file.",
            X = 2,
            Y = Pos.AnchorEnd(1),
            Width = Dim.Fill(2),
        });
        window.Add(page);
        choices.SetFocus();
    }

    private static void ShowMetadata(IApplication app, Window window, GgufFile file, string path)
    {
        string contents;
        if (file.Metadata.Length == 0)
        {
            contents = "No metadata entries.";
        }
        else
        {
            contents = string.Join(Environment.NewLine,
                file.Metadata
                    .Select(m => (m.Key, Value: m.GetValueAsString()))
                    .OrderBy(kvp => kvp.Key)
                    .Select(kvp => $"[{kvp.Key}]: {kvp.Value}"));
        }

        ShowDetails(app, window, "Metadata", contents, file, path);
    }

    private static void ShowTensors(IApplication app, Window window, GgufFile file, string path)
    {
        string contents;
        if (file.TensorsInfos.Length == 0)
        {
            contents = "No tensors found.";
        }
        else
        {
            contents = string.Join(Environment.NewLine,
                file.TensorsInfos
                    .Select(tensor => $"[{tensor.Name}]: {tensor.Type}  [{string.Join(", ", tensor.Dimensions)}]"));
        }

        ShowDetails(app, window, "Tensors", contents, file, path);
    }

    private static void ShowDetails(IApplication app, Window window, string title, string contents, GgufFile file, string path)
    {
        window.RemoveAll();

        var page = new View
        {
            CanFocus = true,
            Width = Dim.Fill(),
            Height = Dim.Fill(),
        };
        var details = new Editor
        {
            Text = contents,
            ReadOnly = true,
            X = 1,
            Y = 1,
            Width = Dim.Fill(1),
            Height = Dim.Fill(2),
            ViewportSettings = ViewportSettingsFlags.HasScrollBars,
        };
        details.KeyDown += (_, key) =>
        {
            if (key.KeyCode == KeyCode.Esc)
            {
                key.Handled = true;
                ShowExplorerMenu(app, window, file, path);
            }
        };
        page.Add(details);
        page.Add(new Label
        {
            Text = $"{title} - Press Esc to return to the selection menu.",
            X = 1,
            Y = Pos.AnchorEnd(1),
            Width = Dim.Fill(1),
        });
        window.Add(page);
        details.SetFocus();
    }

    private static GgufFile? TryLoadGguf(string path, out string? error)
    {
        path = path.Trim('"');
        error = null;
        if (string.IsNullOrWhiteSpace(path))
        {
            error = "A file path is required. Enter the full path to a GGUF file and press Enter.";
            return null;
        }

        if (!File.Exists(path))
        {
            error = $"""File not found: "{path}". Check the path and make sure the file exists.""";
            return null;
        }

        try
        {
            using (var stream = File.OpenRead(path))
            using (var reader = new BinaryReader(stream))
            {
                var magic = new string(reader.ReadChars(4));
                if (magic.Length != 4 || magic != "GGUF")
                {
                    error = $"""
                        Invalid file type: "{path}" is not a GGUF file.
                        Expected the GGUF signature ('GGUF'), but found "{magic}".
                        """;
                    return null;
                }
            }

            return GgufFile.Load(path);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException or OverflowException)
        {
            error = $"""
                The file "{path}" has a GGUF signature but could not be read as a valid GGUF file.
                Details: {exception.Message}
                """;
            return null;
        }
    }

    private static void InsertPastedPath(TextField pathField, string text)
    {
        string firstLine = text.Split(['\r', '\n'], 2)[0];
        string pasteText = new(firstLine.Where(character => !char.IsControl(character)).ToArray());
        pathField.InsertText(pasteText);
    }

    private static string GetConfigPath() =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "NetInfer", "last-gguf-path.txt");

    private static string LoadLastPath(out string? error)
    {
        error = null;
        string configPath = GetConfigPath();

        try
        {
            return File.ReadAllText(configPath).TrimEnd('\r', '\n');
        }
        catch (Exception exception) when (exception is FileNotFoundException or DirectoryNotFoundException)
        {
            return string.Empty;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            error = $"Could not read the saved GGUF path from \"{configPath}\": {exception.Message}";
            return string.Empty;
        }
    }

    private static string? SaveLastPath(string path)
    {
        string configPath = GetConfigPath();
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(configPath)!);
            File.WriteAllText(configPath, path);
            return null;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or NotSupportedException)
        {
            return $"The GGUF file loaded, but its path could not be saved: {exception.Message}";
        }
    }
#endif
}
