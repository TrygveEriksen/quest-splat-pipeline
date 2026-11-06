using UnityEngine;
using UnityEngine.InputSystem;

namespace QuestSplatViewer
{
    /// <summary>
    /// Pick the splat up with the right grip and put it down somewhere else.
    ///
    /// While the grip is held the scene follows the hand, turning only about the
    /// vertical axis so the object stays upright. On release, if a real surface
    /// from Space Setup is under it (see <see cref="SceneSurfaces"/>), the object
    /// is dropped so its lowest point rests on that surface -- a table, say.
    /// </summary>
    public class SplatGrab : MonoBehaviour
    {
        public Transform splatRoot;
        public SplatSceneBrowser browser;
        [Tooltip("Right controller pointer pose.")]
        public Transform hand;
        public SceneSurfaces surfaces;

        public InputAction grip = new InputAction("Grab", InputActionType.Button,
            "<XRController>{RightHand}/{GripButton}");

        Vector3 offset;     // splat root relative to the hand, in the hand's yaw frame
        float yawOffset;
        public bool Held { get; private set; }

        void OnEnable() => grip.Enable();
        void OnDisable() => grip.Disable();

        void Update()
        {
            if (splatRoot == null || hand == null) return;
            if (grip.WasPressedThisFrame()) Begin();
            if (Held) Follow();
            if (Held && grip.WasReleasedThisFrame()) Release();
        }

        public void Begin()
        {
            Quaternion handYaw = Yaw(hand.rotation);
            offset = Quaternion.Inverse(handYaw) * (splatRoot.position - hand.position);
            yawOffset = splatRoot.eulerAngles.y - handYaw.eulerAngles.y;
            Held = true;
        }

        void Follow()
        {
            Quaternion handYaw = Yaw(hand.rotation);
            splatRoot.position = hand.position + handYaw * offset;
            splatRoot.rotation = Quaternion.Euler(0f, handYaw.eulerAngles.y + yawOffset, 0f);
        }

        public void Release()
        {
            Held = false;
            if (surfaces == null || browser == null ||
                !SplatLoaderAdapter.TryGetWorldBounds(browser.Current, out Bounds bounds))
                return;

            var bottom = new Vector3(bounds.center.x, bounds.min.y, bounds.center.z);
            if (surfaces.TryFindSurfaceBelow(bottom, out Vector3 surface, out string label))
            {
                splatRoot.position += Vector3.up * (surface.y - bottom.y);
                Debug.Log($"[Splat] set down on {label} at y={surface.y:F2}");
            }
        }

        static Quaternion Yaw(Quaternion q) => Quaternion.Euler(0f, q.eulerAngles.y, 0f);
    }
}
