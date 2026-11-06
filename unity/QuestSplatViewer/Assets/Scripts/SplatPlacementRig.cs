using UnityEngine;
using UnityEngine.InputSystem;

namespace QuestSplatViewer
{
    /// <summary>
    /// Scale / rotate / raise the loaded scene from the controllers.
    ///
    /// This exists to validate stage C's --scale factor in the headset, which is
    /// the only place a wrong world scale is actually obvious: on a monitor a
    /// dollhouse and a cathedral look the same. Read the logged multiplier back
    /// out and bake it into fixup_ply.py so the shipped file is correct, rather
    /// than leaving a correction factor living in the Unity scene.
    /// </summary>
    public class SplatPlacementRig : MonoBehaviour
    {
        public Transform splatRoot;
        public float scaleRate = 0.6f;      // multiplicative, per second
        public float yawRate = 60f;         // degrees per second
        public float heightRate = 0.5f;     // metres per second

        // Bound by usage rather than control name, so any OpenXR controller
        // profile that reports a primary 2D axis (Touch thumbsticks) works.
        [Header("Input (right stick scales/yaws, left stick raises)")]
        public InputAction scaleYawAction = new InputAction("ScaleYaw", InputActionType.Value,
            "<XRController>{RightHand}/{Primary2DAxis}", expectedControlType: "Vector2");
        public InputAction heightAction = new InputAction("Height", InputActionType.Value,
            "<XRController>{LeftHand}/{Primary2DAxis}", expectedControlType: "Vector2");

        float accumulatedScale = 1f;

        void OnEnable()
        {
            scaleYawAction.Enable();
            heightAction.Enable();
        }

        void OnDisable()
        {
            scaleYawAction.Disable();
            heightAction.Disable();
        }

        void Update()
        {
            if (splatRoot == null) return;

            Vector2 sy = scaleYawAction.ReadValue<Vector2>();
            Vector2 hv = heightAction.ReadValue<Vector2>();

            if (Mathf.Abs(sy.y) > 0.1f)
            {
                float factor = Mathf.Exp(sy.y * scaleRate * Time.deltaTime);
                splatRoot.localScale *= factor;
                accumulatedScale *= factor;
            }

            if (Mathf.Abs(sy.x) > 0.1f)
                splatRoot.Rotate(Vector3.up, sy.x * yawRate * Time.deltaTime, Space.World);

            if (Mathf.Abs(hv.y) > 0.1f)
                splatRoot.position += Vector3.up * (hv.y * heightRate * Time.deltaTime);
        }

        /// <summary>Total scale applied this session. Feed this to --scale.</summary>
        public float AccumulatedScale => accumulatedScale;

        void OnApplicationQuit()
        {
            if (Mathf.Abs(accumulatedScale - 1f) > 0.01f)
                Debug.Log($"[Splat] scene was rescaled by {accumulatedScale:F3}x in-headset. " +
                          $"Re-run: fixup_ply.py --scale <old * {accumulatedScale:F3}>");
        }
    }
}
