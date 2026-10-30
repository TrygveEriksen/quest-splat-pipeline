using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace QuestSplatViewer
{
    /// <summary>
    /// Finds splat files on the headset and loads one at runtime, so each new
    /// capture is an `adb push` instead of an APK rebuild.
    ///
    /// Search order matters. Application.persistentDataPath resolves to
    ///   /sdcard/Android/data/&lt;package&gt;/files/
    /// which adb can write to WITHOUT any storage permission. /sdcard/Download
    /// is checked second because on API 29+ scoped storage it needs
    /// READ_EXTERNAL_STORAGE and requestLegacyExternalStorage, which is a
    /// permission prompt and an extra manifest flag for no real benefit.
    ///
    ///   adb push quest.ply /sdcard/Android/data/&lt;your.package&gt;/files/
    /// </summary>
    public class SplatSceneBrowser : MonoBehaviour
    {
        [Tooltip("Parent for the loaded splat scene. Use this to position the " +
                 "capture relative to the player without touching the splat data.")]
        public Transform splatRoot;

        [Tooltip("Loaded automatically on start if present; otherwise the first " +
                 "file found is used.")]
        public string preferredFileName = "quest.ply";

        static readonly string[] Extensions = { "*.ply", "*.spz", "*.sog" };

        GameObject current;

        public List<string> Discovered { get; } = new List<string>();

        void Start()
        {
            if (splatRoot == null) splatRoot = transform;
            Refresh();
            if (Discovered.Count == 0)
            {
                Debug.LogWarning($"[Splat] no splat files found. Push one with:\n" +
                                 $"  adb push quest.ply {Application.persistentDataPath}/");
                return;
            }

            int index = Discovered.FindIndex(
                p => Path.GetFileName(p).Equals(preferredFileName,
                     System.StringComparison.OrdinalIgnoreCase));
            Load(Discovered[index >= 0 ? index : 0]);
        }

        public void Refresh()
        {
            Discovered.Clear();
            foreach (var dir in SearchDirectories())
            {
                if (!Directory.Exists(dir)) continue;
                foreach (var pattern in Extensions)
                {
                    try { Discovered.AddRange(Directory.GetFiles(dir, pattern)); }
                    catch (IOException) { /* unreadable directory; skip */ }
                }
            }
            Debug.Log($"[Splat] found {Discovered.Count} splat file(s)");
        }

        static IEnumerable<string> SearchDirectories()
        {
            yield return Application.persistentDataPath;   // no permission needed
            yield return "/sdcard/Download";               // needs storage permission
        }

        public void Load(string path)
        {
            if (current != null) Destroy(current);
            current = SplatLoaderAdapter.Load(path, splatRoot);
            if (current != null) LoadedPath = path;
        }

        public string LoadedPath { get; private set; }

        /// <summary>Cycle to the next discovered file, for quick A/B on device.</summary>
        public void LoadNext()
        {
            if (Discovered.Count == 0) return;
            int i = Mathf.Max(0, Discovered.IndexOf(LoadedPath));
            Load(Discovered[(i + 1) % Discovered.Count]);
        }
    }
}
