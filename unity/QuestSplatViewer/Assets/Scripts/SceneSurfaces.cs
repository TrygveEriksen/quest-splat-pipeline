using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
#if UNITY_ANDROID && !UNITY_EDITOR
using UnityEngine.Android;
#endif

namespace QuestSplatViewer
{
    /// <summary>
    /// Real surfaces from the headset's Space Setup -- tables, floor, couches --
    /// as AR Foundation planes, so a splat can be set down on a real table.
    ///
    /// Meta's runtime doesn't detect planes live: it returns what the user drew
    /// in Space Setup (on a Quest 2, marked by hand). Reading it needs the
    /// com.oculus.permission.USE_SCENE runtime permission, so the plane manager
    /// stays disabled until that is granted, as the Meta OpenXR docs require.
    /// </summary>
    public class SceneSurfaces : MonoBehaviour
    {
        const string ScenePermission = "com.oculus.permission.USE_SCENE";

        public ARPlaneManager planes;
        [Tooltip("How far below a point to look for a surface, in metres.")]
        public float maxDrop = 0.75f;

        public int Count => planes != null && planes.enabled ? planes.trackables.count : 0;

        void Start()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (Permission.HasUserAuthorizedPermission(ScenePermission))
            {
                Enable();
                return;
            }
            var callbacks = new PermissionCallbacks();
            callbacks.PermissionGranted += _ => Enable();
            callbacks.PermissionDenied += _ =>
                Debug.LogWarning("[Surfaces] scene permission denied: splats won't snap to tables");
            Permission.RequestUserPermission(ScenePermission, callbacks);
#else
            Enable();
#endif
        }

        void Enable()
        {
            if (planes == null) return;
            planes.trackablesChanged.AddListener(OnPlanesChanged);
            planes.enabled = true;
        }

        void OnPlanesChanged(ARTrackablesChangedEventArgs<ARPlane> changes)
        {
            if (changes.added.Count == 0) return;
            var labels = new List<string>();
            foreach (var plane in planes.trackables)
                labels.Add($"{plane.classifications} ({plane.alignment}, y={plane.center.y:F2})");
            Debug.Log($"[Surfaces] {planes.trackables.count} plane(s): {string.Join(", ", labels)}");
        }

        /// <summary>
        /// The highest upward-facing surface under <paramref name="point"/>, within
        /// <see cref="maxDrop"/> -- so a table wins over the floor beneath it.
        /// </summary>
        public bool TryFindSurfaceBelow(Vector3 point, out Vector3 surfacePoint, out string label)
        {
            surfacePoint = default;
            label = null;
            if (Count == 0) return false;

            float best = float.NegativeInfinity;
            foreach (var plane in planes.trackables)
            {
                if (plane.alignment != PlaneAlignment.HorizontalUp) continue;
                float y = plane.center.y;
                // A little above counts too: released slightly sunk into a table
                // should still come to rest on top of it.
                if (y > point.y + 0.15f || point.y - y > maxDrop || y <= best) continue;

                // The boundary is in plane space: (x, y) there is local (x, 0, z).
                Vector3 local = plane.transform.InverseTransformPoint(new Vector3(point.x, y, point.z));
                if (!Inside(plane.boundary, new Vector2(local.x, local.z))) continue;

                best = y;
                surfacePoint = new Vector3(point.x, y, point.z);
                label = plane.classifications.ToString();
            }
            return !float.IsNegativeInfinity(best);
        }

        static bool Inside(NativeArray<Vector2> polygon, Vector2 p)
        {
            bool inside = false;
            for (int i = 0, j = polygon.Length - 1; i < polygon.Length; j = i++)
            {
                Vector2 a = polygon[i], b = polygon[j];
                if ((a.y > p.y) != (b.y > p.y) &&
                    p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)
                    inside = !inside;
            }
            return inside;
        }
    }
}
