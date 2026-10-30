using UnityEngine;
using UnityEngine.XR;

namespace QuestSplatViewer
{
    /// <summary>
    /// Wrist-mounted frame-time readout.
    ///
    /// Non-negotiable for this project: the whole Quest 2 splat budget has to be
    /// found by measurement, because no published Adreno 650 splat benchmarks
    /// exist. Guessing from desktop framerate will mislead you.
    ///
    /// Reports the 95th-percentile frame time as well as the mean -- splat
    /// sorting causes periodic spikes that a mean hides, and spikes are what
    /// actually read as judder in a headset.
    /// </summary>
    public class PerfHud : MonoBehaviour
    {
        public TextMesh target;
        public SplatSceneBrowser browser;
        public float refreshInterval = 0.5f;

        const int Window = 180;
        readonly float[] samples = new float[Window];
        int count, cursor;
        float nextRefresh;

        void Update()
        {
            samples[cursor] = Time.unscaledDeltaTime * 1000f;
            cursor = (cursor + 1) % Window;
            count = Mathf.Min(count + 1, Window);

            if (Time.unscaledTime < nextRefresh || target == null) return;
            nextRefresh = Time.unscaledTime + refreshInterval;

            var sorted = new float[count];
            System.Array.Copy(samples, sorted, count);
            System.Array.Sort(sorted);

            float mean = 0f;
            for (int i = 0; i < count; i++) mean += sorted[i];
            mean /= count;
            float p95 = sorted[Mathf.Clamp((int)(count * 0.95f), 0, count - 1)];

            float refresh = XRDevice.refreshRate > 1f ? XRDevice.refreshRate : 72f;
            float budget = 1000f / refresh;
            string verdict = p95 <= budget ? "OK" : "OVER BUDGET";

            string file = browser != null && !string.IsNullOrEmpty(browser.LoadedPath)
                ? System.IO.Path.GetFileName(browser.LoadedPath)
                : "(none)";

            target.text =
                $"{file}\n" +
                $"mean {mean:F1} ms   p95 {p95:F1} ms\n" +
                $"{1000f / Mathf.Max(mean, 0.01f):F0} fps  /  {refresh:F0} Hz target\n" +
                $"budget {budget:F1} ms  [{verdict}]";
            target.color = p95 <= budget ? Color.green
                         : p95 <= budget * 1.25f ? Color.yellow : Color.red;
        }
    }
}
