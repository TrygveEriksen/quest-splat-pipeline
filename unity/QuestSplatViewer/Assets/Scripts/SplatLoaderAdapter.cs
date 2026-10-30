using System;
using System.IO;
using UnityEngine;

namespace QuestSplatViewer
{
    /// <summary>
    /// The ONLY place that touches the UnitySplats package API.
    ///
    /// That API surface is taken from the package README rather than verified
    /// against a compiled project, so if the package has moved on, this is the
    /// single file to fix -- nothing else in the app references the package.
    ///
    /// Expected upstream shape (arloopa/UnitySplats, Unity 6):
    ///     GsplatRuntimeLoader.LoadFile(string absolutePath, CompressionMode mode)
    /// returning a component/handle that can be parented into the scene.
    /// </summary>
    public static class SplatLoaderAdapter
    {
        /// <summary>
        /// Load a .ply/.spz splat file and return the GameObject holding it.
        /// Returns null and logs on failure; callers must handle null.
        /// </summary>
        public static GameObject Load(string absolutePath, Transform parent)
        {
            if (!File.Exists(absolutePath))
            {
                Debug.LogError($"[Splat] file not found: {absolutePath}");
                return null;
            }

            try
            {
                // ---------------------------------------------------------
                // ADAPT HERE if the package API differs. Reflection keeps the
                // app compiling even before the package is installed, so you
                // can build and deploy the shell first and wire the renderer
                // once you've confirmed the real method signature.
                // ---------------------------------------------------------
                var loaderType = Type.GetType("GsplatRuntimeLoader")
                                 ?? Type.GetType("UnitySplats.GsplatRuntimeLoader");
                if (loaderType == null)
                {
                    Debug.LogError("[Splat] GsplatRuntimeLoader not found. Install the " +
                                   "UnitySplats package, then confirm the type name here.");
                    return null;
                }

                var method = loaderType.GetMethod("LoadFile",
                    new[] { typeof(string) }) ?? loaderType.GetMethod("LoadFile");
                if (method == null)
                {
                    Debug.LogError("[Splat] GsplatRuntimeLoader.LoadFile not found.");
                    return null;
                }

                var parameters = method.GetParameters();
                object result = parameters.Length == 1
                    ? method.Invoke(null, new object[] { absolutePath })
                    : method.Invoke(null, new object[] { absolutePath, null });

                var go = result as GameObject ?? (result as Component)?.gameObject;
                if (go == null)
                {
                    Debug.LogError($"[Splat] loader returned {result?.GetType().Name ?? "null"}, " +
                                   "which is not a GameObject or Component.");
                    return null;
                }

                go.transform.SetParent(parent, worldPositionStays: false);
                Debug.Log($"[Splat] loaded {Path.GetFileName(absolutePath)}");
                return go;
            }
            catch (Exception e)
            {
                Debug.LogError($"[Splat] load failed for {absolutePath}: {e}");
                return null;
            }
        }
    }
}
