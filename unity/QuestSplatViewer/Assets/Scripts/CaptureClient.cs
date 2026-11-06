using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Networking;

namespace QuestSplatViewer
{
    /// <summary>
    /// The headset side of scripts/splat_server.py. Finds the server on the LAN,
    /// shows frame 0 of a new phone upload on a panel, lets the user mark the
    /// object with the right controller, sends that to the server, and loads the
    /// gaussians when they come back.
    ///
    ///   Right trigger  click on the object (a point) or drag around it (a box)
    ///   A              send the selection
    ///   B              clear it -- or, when nothing is being picked, pick the
    ///                  last capture again (only SAM 2 onward re-runs)
    /// </summary>
    public class CaptureClient : MonoBehaviour
    {
        [Header("Scene")]
        public SplatSceneBrowser browser;
        [Tooltip("A 1x1 quad facing the user; scaled to the frame's aspect ratio.")]
        public Transform panel;
        public Renderer panelRenderer;
        public TextMesh status;
        [Tooltip("Right controller pointer pose.")]
        public Transform aim;
        public LineRenderer ray;
        public Transform pointMarker;
        public LineRenderer boxMarker;

        [Header("Server")]
        [Tooltip("Empty: find the server by UDP broadcast. A server.txt in " +
                 "persistentDataPath (e.g. http://192.168.1.20:8765) overrides both.")]
        public string serverUrl = "";
        public int discoveryPort = 8766;
        public float pollInterval = 1.5f;
        public float panelHeight = 0.6f;

        [Header("Input (right controller)")]
        public InputAction trigger = new InputAction("Pick", InputActionType.Button,
            "<XRController>{RightHand}/{TriggerButton}");
        public InputAction send = new InputAction("Send", InputActionType.Button,
            "<XRController>{RightHand}/{PrimaryButton}");
        public InputAction clear = new InputAction("Clear", InputActionType.Button,
            "<XRController>{RightHand}/{SecondaryButton}");

        // Field names mirror the server's status JSON (snake_case on purpose).
        [Serializable]
        public class Job
        {
            public string id, name, state, message, error, inference, frame_url, result_url;
            public int frames, splats;
        }

        [Serializable]
        class JobList { public Job[] items; }

        enum Mode { Idle, Picking, Waiting, Downloading }

        Mode mode = Mode.Idle;
        Job latest;                 // newest job on the server
        string pickJob;             // job whose frame is on the panel
        string waitJob;             // job we sent a selection for
        Vector2Int frameSize;
        Vector2? dragStart, dragEnd;
        string pendingPrompt;       // JSON body, once something is selected
        int failures;
        bool holdStatus;            // keep a result/error message up while idle

        public string Status { get; private set; } = "";

        void OnEnable()
        {
            trigger.Enable();
            send.Enable();
            clear.Enable();
        }

        void OnDisable()
        {
            trigger.Disable();
            send.Disable();
            clear.Disable();
        }

        void Start()
        {
            ShowPanel(false);
            StartCoroutine(Run());
        }

        // ------------------------------------------------------------------
        // Server discovery and polling
        // ------------------------------------------------------------------
        IEnumerator Run()
        {
            yield return Discover();
            while (true)
            {
                if (mode != Mode.Downloading)
                    yield return Poll();
                yield return new WaitForSeconds(pollInterval);
            }
        }

        IEnumerator Discover()
        {
            string file = Path.Combine(Application.persistentDataPath, "server.txt");
            if (File.Exists(file))
                serverUrl = File.ReadAllText(file).Trim();
            if (!string.IsNullOrEmpty(serverUrl))
            {
                serverUrl = serverUrl.TrimEnd('/');
                Debug.Log($"[Capture] server {serverUrl} (configured)");
                yield break;
            }

            byte[] request = Encoding.ASCII.GetBytes("QSV_DISCOVER");
            using var udp = new UdpClient();
            udp.EnableBroadcast = true;
            udp.Client.Blocking = false;
            while (string.IsNullOrEmpty(serverUrl))
            {
                SetStatus("Looking for the capture server on this network...");
                try
                {
                    udp.Send(request, request.Length, new IPEndPoint(IPAddress.Broadcast, discoveryPort));
                }
                catch (SocketException e)
                {
                    Debug.LogWarning($"[Capture] discovery broadcast failed: {e.Message}");
                }

                float until = Time.realtimeSinceStartup + 2f;
                while (Time.realtimeSinceStartup < until && string.IsNullOrEmpty(serverUrl))
                {
                    while (udp.Available > 0)
                    {
                        IPEndPoint from = null;
                        string[] reply = Encoding.ASCII.GetString(udp.Receive(ref from)).Split(' ');
                        if (reply.Length == 2 && reply[0] == "QSV_SERVER")
                            serverUrl = $"http://{from.Address}:{reply[1]}";
                    }
                    yield return null;
                }
            }
            Debug.Log($"[Capture] server {serverUrl} (discovered)");
        }

        IEnumerator Poll()
        {
            using var req = UnityWebRequest.Get(serverUrl + "/api/jobs");
            req.timeout = 5;
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                if (++failures == 3)
                    SetStatus($"Can't reach the capture server at {serverUrl}\n({req.error})");
                yield break;
            }
            failures = 0;

            var jobs = JsonUtility.FromJson<JobList>("{\"items\":" + req.downloadHandler.text + "}").items;
            latest = jobs.Length > 0 ? jobs[0] : null;
            Job Find(string id) => id == null ? null : Array.Find(jobs, j => j.id == id);

            switch (mode)
            {
                case Mode.Idle:
                    // Finished jobs from before launch aren't loaded: the scene
                    // starts empty, and B brings the last one back to pick again.
                    if (latest?.state == "awaiting_prompt")
                        yield return StartPicking(latest);
                    else if (latest != null && latest.state != "done" && latest.state != "failed")
                        SetStatus($"{latest.name}: {Describe(latest)}");
                    else if (!holdStatus)
                        SetStatus($"Upload a video from your phone at\n{serverUrl}/" +
                                  (latest?.state == "done" ? "\nB: pick the last capture again" : ""));
                    break;

                case Mode.Picking:
                    var picking = Find(pickJob);
                    if (latest != null && latest.id != pickJob && latest.state == "awaiting_prompt")
                        yield return StartPicking(latest);       // a newer upload wins
                    else if (picking == null || picking.state == "failed")
                    {
                        ShowPanel(false);
                        mode = Mode.Idle;
                        SetStatus(picking?.error ?? "That capture is gone from the server.");
                    }
                    break;

                case Mode.Waiting:
                    var waiting = Find(waitJob);
                    if (waiting == null || waiting.state == "failed")
                    {
                        mode = Mode.Idle;
                        holdStatus = true;
                        SetStatus($"Failed: {waiting?.error ?? "job missing on the server"}");
                    }
                    else if (waiting.state == "done" && !string.IsNullOrEmpty(waiting.result_url))
                        yield return Download(waiting);
                    else
                        SetStatus($"{waiting.name}: {Describe(waiting)}");
                    break;
            }
        }

        static string Describe(Job job) =>
            string.IsNullOrEmpty(job.message) ? job.state.Replace('_', ' ') : job.message;

        static string LocalPath(Job job) =>
            Path.Combine(Application.persistentDataPath, $"capture-{job.id}.ply");

        // ------------------------------------------------------------------
        // Picking
        // ------------------------------------------------------------------
        IEnumerator StartPicking(Job job)
        {
            using var req = UnityWebRequestTexture.GetTexture(serverUrl + job.frame_url);
            req.timeout = 15;
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                SetStatus($"Couldn't download the frame: {req.error}");
                yield break;
            }

            var tex = DownloadHandlerTexture.GetContent(req);
            frameSize = new Vector2Int(tex.width, tex.height);
            var old = panelRenderer.material.mainTexture;
            panelRenderer.material.mainTexture = tex;
            if (old != null && old != tex) Destroy(old);
            panel.localScale = new Vector3(panelHeight * tex.width / tex.height, panelHeight, 1f);

            pickJob = job.id;
            mode = Mode.Picking;
            holdStatus = false;
            ClearPick();
            ShowPanel(true);
            Debug.Log($"[Capture] picking {job.name} ({tex.width}x{tex.height})");
        }

        void Update()
        {
            bool picking = mode == Mode.Picking;
            if (ray != null) ray.enabled = picking && aim != null;

            if (!picking)
            {
                if (mode == Mode.Idle && clear.WasPressedThisFrame())
                    StartCoroutine(PickAgain());
                return;
            }

            if (aim != null)
            {
                var r = new Ray(aim.position, aim.forward);
                bool hit = TryPanelHit(r, out Vector2 px, out Vector3 world);
                ray.SetPosition(0, r.origin);
                ray.SetPosition(1, hit ? world : r.origin + r.direction * 3f);

                if (hit && trigger.WasPressedThisFrame())
                    dragStart = dragEnd = px;
                else if (dragStart.HasValue && hit && trigger.IsPressed())
                    dragEnd = px;
                if (dragStart.HasValue && trigger.WasReleasedThisFrame())
                {
                    Vector2 a = dragStart.Value, b = dragEnd.Value;
                    if (Vector2.Distance(a, b) < 0.03f * frameSize.y) PickPoint(a);
                    else PickBox(a, b);
                }
            }

            if (send.WasPressedThisFrame()) Send();
            if (clear.WasPressedThisFrame()) ClearPick();
            DrawSelection();
        }

        /// <summary>Intersect a ray with the panel; pixel is in frame coordinates
        /// (origin top-left, as SAM 2 expects).</summary>
        public bool TryPanelHit(Ray worldRay, out Vector2 pixel, out Vector3 world)
        {
            pixel = default;
            world = default;
            if (!panel.gameObject.activeInHierarchy) return false;
            // InverseTransformVector, not ...Direction: the panel is scaled to the
            // frame's aspect, and the direction must go through that scale too.
            Vector3 o = panel.InverseTransformPoint(worldRay.origin);
            Vector3 d = panel.InverseTransformVector(worldRay.direction);
            if (Mathf.Abs(d.z) < 1e-6f) return false;
            float t = -o.z / d.z;
            if (t <= 0f) return false;
            Vector3 p = o + t * d;
            if (Mathf.Abs(p.x) > 0.5f || Mathf.Abs(p.y) > 0.5f) return false;
            world = panel.TransformPoint(p);
            pixel = new Vector2((p.x + 0.5f) * frameSize.x, (0.5f - p.y) * frameSize.y);
            return true;
        }

        Vector3 PixelToWorld(Vector2 px) => panel.TransformPoint(new Vector3(
            px.x / frameSize.x - 0.5f, 0.5f - px.y / frameSize.y, -0.003f));

        public void PickPoint(Vector2 px)
        {
            dragStart = dragEnd = px;
            pendingPrompt = $"{{\"points\":[[{F(px.x)},{F(px.y)}]]}}";
            SetStatus("Point set. A: send   B: clear\n(or click again / drag a box)");
        }

        public void PickBox(Vector2 a, Vector2 b)
        {
            Vector2 min = Vector2.Min(a, b), max = Vector2.Max(a, b);
            dragStart = min;
            dragEnd = max;
            pendingPrompt = $"{{\"box\":[{F(min.x)},{F(min.y)},{F(max.x)},{F(max.y)}]}}";
            SetStatus("Box set. A: send   B: clear\n(or drag again / click a point)");
        }

        void ClearPick()
        {
            dragStart = dragEnd = null;
            pendingPrompt = null;
            SetStatus("Point at the object with the right controller.\n" +
                      "Trigger: click it, or drag a box around it.");
        }

        void DrawSelection()
        {
            bool isBox = pendingPrompt != null && pendingPrompt.Contains("box");
            bool dragging = dragStart.HasValue && trigger.IsPressed();
            bool showBox = dragStart.HasValue && (isBox || dragging) && dragStart != dragEnd;
            bool showPoint = dragStart.HasValue && !showBox;

            pointMarker.gameObject.SetActive(showPoint);
            if (showPoint) pointMarker.position = PixelToWorld(dragStart.Value);

            boxMarker.enabled = showBox;
            if (showBox)
            {
                Vector2 a = dragStart.Value, b = dragEnd.Value;
                boxMarker.SetPosition(0, PixelToWorld(a));
                boxMarker.SetPosition(1, PixelToWorld(new Vector2(b.x, a.y)));
                boxMarker.SetPosition(2, PixelToWorld(b));
                boxMarker.SetPosition(3, PixelToWorld(new Vector2(a.x, b.y)));
            }
        }

        public void Send()
        {
            if (mode == Mode.Picking && pendingPrompt != null)
                StartCoroutine(SendPrompt(pickJob, pendingPrompt));
        }

        IEnumerator SendPrompt(string jobId, string body)
        {
            using var req = new UnityWebRequest(serverUrl + $"/api/jobs/{jobId}/prompt", "POST");
            req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            req.timeout = 10;
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                SetStatus($"Sending failed: {req.error}\n{req.downloadHandler.text}");
                yield break;
            }
            Debug.Log($"[Capture] sent {body} for {jobId}");
            waitJob = jobId;
            mode = Mode.Waiting;
            ShowPanel(false);
            SetStatus("Sent. Reconstructing...");
        }

        IEnumerator PickAgain()
        {
            if (latest != null && latest.state == "done")
                yield return StartPicking(latest);
        }

        // ------------------------------------------------------------------
        // Result
        // ------------------------------------------------------------------
        IEnumerator Download(Job job)
        {
            mode = Mode.Downloading;
            string path = LocalPath(job), part = path + ".part";
            SetStatus($"{job.name}: downloading {job.splats:N0} splats...");
            using (var req = UnityWebRequest.Get(serverUrl + job.result_url))
            {
                req.downloadHandler = new DownloadHandlerFile(part) { removeFileOnAbort = true };
                req.timeout = 120;
                yield return req.SendWebRequest();
                if (req.result != UnityWebRequest.Result.Success)
                {
                    mode = Mode.Idle;
                    SetStatus($"Download failed: {req.error}");
                    yield break;
                }
            }
            if (File.Exists(path)) File.Delete(path);
            File.Move(part, path);

            browser.Refresh();
            browser.Load(path);
            mode = Mode.Idle;
            holdStatus = true;
            SetStatus($"{job.name}: {job.splats:N0} splats\nB: pick the object again");
            Debug.Log($"[Capture] loaded {job.id} from {path}");
        }

        // ------------------------------------------------------------------
        void ShowPanel(bool on)
        {
            panel.gameObject.SetActive(on);
            if (!on)
            {
                pointMarker.gameObject.SetActive(false);
                boxMarker.enabled = false;
            }
        }

        void SetStatus(string text)
        {
            if (text == Status) return;
            Status = text;
            if (status != null) status.text = text;
            Debug.Log("[Capture] " + text.Replace('\n', ' '));
        }

        static string F(float v) => v.ToString("F1", CultureInfo.InvariantCulture);
    }
}
