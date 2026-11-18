using System;
using System.IO;
using Gsplat;
using UnityEngine;

namespace QuestSplatViewer
{
    /// <summary>
    /// The ONLY place that touches the UnitySplats package API
    /// (com.arloopa.unitysplats, pinned to a commit in Packages/manifest.json).
    ///
    /// GsplatRuntimeLoader.LoadFile decodes a .ply/.spz/.sog into an in-memory
    /// GsplatAsset; a GsplatRenderer on a GameObject draws it. The asset is not
    /// owned by the GameObject, so it has to be destroyed explicitly -- use
    /// <see cref="Unload"/> rather than Destroy on the returned object.
    /// </summary>
    public static class SplatLoaderAdapter
    {
        /// <summary>
        /// The frame splat_server.py writes PLYs in: standard 3DGS / OpenCV, x right,
        /// y down, z forward -- the same file previews upright in SuperSplat. The
        /// package converts it to Unity's x right, y up, z forward on load, for
        /// positions, rotations and SH alike, so no flip is baked into the file.
        /// </summary>
        const SourceCoordinates PlyFrame = SourceCoordinates.RDF;

        /// <summary>
        /// Load a .ply/.spz/.sog splat file and return the GameObject drawing it.
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
                // Spark is the package's packed format: far less GPU memory and
                // upload time than Uncompressed, which matters on a Quest.
                GsplatAsset asset = GsplatRuntimeLoader.LoadFile(
                    absolutePath, CompressionMode.Spark, PlyFrame);

                var go = new GameObject(Path.GetFileNameWithoutExtension(absolutePath));
                go.transform.SetParent(parent, worldPositionStays: false);
                go.AddComponent<GsplatRenderer>().GsplatAsset = asset;

                Debug.Log($"[Splat] loaded {Path.GetFileName(absolutePath)}: " +
                          $"{asset.SplatCount:N0} splats");
                return go;
            }
            catch (Exception e)
            {
                Debug.LogError($"[Splat] load failed for {absolutePath}: {e}");
                return null;
            }
        }

        /// <summary>World-space box around a loaded splat, from its asset's bounds.</summary>
        public static bool TryGetWorldBounds(GameObject splat, out Bounds bounds)
        {
            bounds = default;
            var renderer = splat != null ? splat.GetComponent<GsplatRenderer>() : null;
            if (renderer == null || renderer.GsplatAsset == null) return false;

            Bounds local = renderer.GsplatAsset.Bounds;
            for (int i = 0; i < 8; i++)
            {
                var corner = local.center + Vector3.Scale(local.extents, new Vector3(
                    (i & 1) == 0 ? -1 : 1, (i & 2) == 0 ? -1 : 1, (i & 4) == 0 ? -1 : 1));
                var world = splat.transform.TransformPoint(corner);
                if (i == 0) bounds = new Bounds(world, Vector3.zero);
                else bounds.Encapsulate(world);
            }
            return true;
        }

        /// <summary>Destroy a splat loaded by <see cref="Load"/>, and its asset.</summary>
        public static void Unload(GameObject splat)
        {
            if (splat == null) return;
            var renderer = splat.GetComponent<GsplatRenderer>();
            if (renderer != null && renderer.GsplatAsset != null)
            {
                var asset = renderer.GsplatAsset;
                renderer.GsplatAsset = null;
                UnityEngine.Object.Destroy(asset);
            }
            UnityEngine.Object.Destroy(splat);
        }
    }
}
