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

        [Header("Input (right stick scales/yaws, left stick raises)")]
        public InputActionReference scaleYawAction;
        public InputActionReference heightAction;

        float accumulatedScale = 1f;

        void OnEnable()
        {
            scaleYawAction?.action?.Enable();
            heightAction?.action?.Enable();
        }

        void OnDisable()
        {
            scaleYawAction?.action?.Disable();
            heightAction?.action?.Disable();
        }

        void Update()
        {
            if (splatRoot == null) return;

            Vector2 sy = scaleYawAction?.action?.ReadValue<Vector2>() ?? Vector2.zero;
            Vector2 hv = heightAction?.action?.ReadValue<Vector2>() ?? Vector2.zero;

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
