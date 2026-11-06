using System;
using System.IO;
using System.Linq;
using System.Text;
using Unity.XR.CoreUtils;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEditor.XR.Management;
using UnityEditor.XR.Management.Metadata;
using UnityEditor.XR.OpenXR.Features;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.XR;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.SceneManagement;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR;
using UnityEngine.XR.OpenXR.Features;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.OpenXR.Features.Interactions;
using UnityEngine.XR.OpenXR.Features.Meta;
using UnityEngine.XR.OpenXR.Features.MetaQuestSupport;

namespace QuestSplatViewer.EditorTools
{
    /// <summary>
    /// Applies the settings docs/UNITY_SETUP.md describes and builds the viewer
    /// scene, so the project can be set up and built from the command line:
    ///
    ///   unity command eval "return QuestSplatViewer.EditorTools.QuestProjectSetup.Apply();"
    ///   unity command eval "return QuestSplatViewer.EditorTools.QuestProjectSetup.BuildApk();"
    ///
    /// Idempotent: re-run it after changing anything here. The scene is rebuilt
    /// from scratch each time, so edit this file rather than the scene.
    /// </summary>
    public static class QuestProjectSetup
    {
        public const string ScenePath = "Assets/Scenes/SplatViewer.unity";
        public const string ApplicationId = "com.vrproj.questsplatviewer";
        public const string DefaultApkPath = "Builds/QuestSplatViewer.apk";

        const string MobilePipeline = "Assets/Settings/Mobile_RPAsset.asset";   // Android's quality level
        static readonly string[] Renderers =
        {
            "Assets/Settings/Mobile_Renderer.asset",
            "Assets/Settings/PC_Renderer.asset",      // the Editor's, for play-mode checks
        };

        // Where the splat appears relative to the floor-level tracking origin:
        // a metre in front, about table height.
        static readonly Vector3 SplatPosition = new Vector3(0f, 1.0f, 1.0f);
        // The capture panel (frame 0 of a phone upload), just in front of the splat.
        static readonly Vector3 PanelPosition = new Vector3(0f, 1.35f, 0.8f);
        const string MaterialDir = "Assets/Materials";

        [MenuItem("Quest Splat Viewer/Apply Project Setup")]
        public static string Apply()
        {
            var log = new StringBuilder();
            ConfigurePlayer(log);
            ConfigureXR(log);
            ConfigureUrp(log);
            BuildScene(log);
            AssetDatabase.SaveAssets();
            Debug.Log("[Setup]\n" + log);
            return log.ToString();
        }

        [MenuItem("Quest Splat Viewer/Build APK")]
        public static void BuildApkMenu() => BuildApk();

        public static string BuildApk(string apkPath = DefaultApkPath)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(apkPath));
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = apkPath,
                target = BuildTarget.Android,
                targetGroup = BuildTargetGroup.Android,
                options = BuildOptions.None,
            });
            var s = report.summary;
            float apkMb = File.Exists(apkPath) ? new FileInfo(apkPath).Length / (1024f * 1024f) : 0f;
            string result = $"{s.result}: {apkPath}, {apkMb:F1} MB, " +
                            $"{s.totalTime.TotalSeconds:F0} s, {s.totalErrors} errors";
            Debug.Log("[Setup] build " + result);
            return result;
        }

        // ------------------------------------------------------------------
        // Player: IL2CPP/ARM64/Vulkan are what Quest needs; Vulkan is also
        // required for UnitySplats' GPU radix sort (GLES falls back to CPU).
        // ------------------------------------------------------------------
        static void ConfigurePlayer(StringBuilder log)
        {
            var android = NamedBuildTarget.Android;
            PlayerSettings.companyName = "vrproj";
            PlayerSettings.productName = "Quest Splat Viewer";
            PlayerSettings.SetApplicationIdentifier(android, ApplicationId);
            PlayerSettings.SetScriptingBackend(android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
            PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.Vulkan });
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel32;
            PlayerSettings.colorSpace = ColorSpace.Linear;
            PlayerSettings.SetMobileMTRendering(android, true);
            // CaptureClient talks plain HTTP to splat_server.py on the LAN, which
            // Android blocks by default.
            PlayerSettings.Android.forceInternetPermission = true;
            PlayerSettings.insecureHttpOption = InsecureHttpOption.AlwaysAllowed;
            log.AppendLine($"player: {ApplicationId}, IL2CPP, ARM64, Vulkan, min API 32, linear, " +
                           "internet + LAN http");
        }

        // ------------------------------------------------------------------
        // XR: OpenXR loader for Android, Meta Quest support, Touch controllers,
        // single-pass instanced (the only XR mode UnitySplats supports on URP).
        // ------------------------------------------------------------------
        static void ConfigureXR(StringBuilder log)
        {
            if (!EditorBuildSettings.TryGetConfigObject(XRGeneralSettings.settingsKey,
                    out XRGeneralSettingsPerBuildTarget perTarget) || perTarget == null)
            {
                Directory.CreateDirectory("Assets/XR");
                perTarget = ScriptableObject.CreateInstance<XRGeneralSettingsPerBuildTarget>();
                AssetDatabase.CreateAsset(perTarget, "Assets/XR/XRGeneralSettingsPerBuildTarget.asset");
                EditorBuildSettings.AddConfigObject(XRGeneralSettings.settingsKey, perTarget, true);
            }
            if (!perTarget.HasSettingsForBuildTarget(BuildTargetGroup.Android))
                perTarget.CreateDefaultSettingsForBuildTarget(BuildTargetGroup.Android);
            if (!perTarget.HasManagerSettingsForBuildTarget(BuildTargetGroup.Android))
                perTarget.CreateDefaultManagerSettingsForBuildTarget(BuildTargetGroup.Android);

            var general = perTarget.SettingsForBuildTarget(BuildTargetGroup.Android);
            general.InitManagerOnStart = true;
            bool assigned = XRPackageMetadataStore.AssignLoader(
                general.Manager, typeof(OpenXRLoader).FullName, BuildTargetGroup.Android);
            EditorUtility.SetDirty(general);
            EditorUtility.SetDirty(perTarget);
            log.AppendLine($"xr: OpenXR loader on Android ({(assigned ? "assigned" : "already assigned")})");

            FeatureHelpers.RefreshFeatures(BuildTargetGroup.Android);
            var openXR = OpenXRSettings.GetSettingsForBuildTargetGroup(BuildTargetGroup.Android);
            openXR.renderMode = OpenXRSettings.RenderMode.SinglePassInstanced;
            // Meta's recommendation (flagged by OpenXR project validation).
            openXR.latencyOptimization = OpenXRSettings.LatencyOptimization.PrioritizeInputPolling;
            Enable<MetaQuestFeature>(openXR, log);
            Enable<OculusTouchControllerProfile>(openXR, log);
            // Mixed reality: passthrough, Space Setup planes, and the session both need.
            Enable<ARSessionFeature>(openXR, log);
            // Passthrough is submitted as a composition layer; without this the
            // runtime gets no layer and the background stays black.
            Enable<UnityEngine.XR.OpenXR.Features.CompositionLayers.OpenXRCompositionLayersFeature>(openXR, log);
            Enable<ARCameraFeature>(openXR, log);
            Enable<ARPlaneFeature>(openXR, log);
            EditorUtility.SetDirty(openXR);
            log.AppendLine("openxr: single pass instanced");
        }

        static void Enable<T>(OpenXRSettings settings, StringBuilder log) where T : OpenXRFeature
        {
            var feature = settings.GetFeature<T>();
            if (feature == null)
                throw new InvalidOperationException($"OpenXR feature {typeof(T).Name} not found");
            feature.enabled = true;
            EditorUtility.SetDirty(feature);
            log.AppendLine($"openxr: enabled {typeof(T).Name}");
        }

        // ------------------------------------------------------------------
        // URP: the splat pass, plus the Quest settings from UNITY_SETUP.md.
        // ------------------------------------------------------------------
        static void ConfigureUrp(StringBuilder log)
        {
            var pipeline = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>(MobilePipeline);
            pipeline.msaaSampleCount = 1;   // costly, and does little for alpha-blended splats
            pipeline.supportsHDR = false;
            var so = new SerializedObject(pipeline);
            SetBool(so, "m_MainLightShadowsSupported", false);   // splats are unlit
            SetBool(so, "m_AdditionalLightShadowsSupported", false);
            SetBool(so, "m_SupportsTerrainHoles", false);        // Meta's passthrough guidance
            so.ApplyModifiedProperties();
            EditorUtility.SetDirty(pipeline);
            log.AppendLine("urp: mobile asset MSAA off, HDR off, shadows off, terrain holes off");

            // The feature class is internal to the package, hence the lookup by name.
            var featureType = Type.GetType("Gsplat.GsplatURPFeature, Gsplat", throwOnError: true);
            foreach (var path in Renderers)
            {
                var data = AssetDatabase.LoadAssetAtPath<UniversalRendererData>(path);
                if (data.rendererFeatures.Any(f => f != null && f.GetType() == featureType))
                {
                    log.AppendLine($"urp: {Path.GetFileName(path)} already has the Gsplat feature");
                    continue;
                }

                var feature = (ScriptableRendererFeature)ScriptableObject.CreateInstance(featureType);
                feature.name = "GsplatURPFeature";
                AssetDatabase.AddObjectToAsset(feature, data);
                AssetDatabase.TryGetGUIDAndLocalFileIdentifier(feature, out _, out long localId);

                // Mirror what the renderer inspector's "Add Renderer Feature" does:
                // the feature list and its local-id map must stay in step.
                var rso = new SerializedObject(data);
                var list = rso.FindProperty("m_RendererFeatures");
                list.arraySize++;
                list.GetArrayElementAtIndex(list.arraySize - 1).objectReferenceValue = feature;
                var map = rso.FindProperty("m_RendererFeatureMap");
                map.arraySize++;
                map.GetArrayElementAtIndex(map.arraySize - 1).longValue = localId;
                rso.ApplyModifiedProperties();
                EditorUtility.SetDirty(data);
                log.AppendLine($"urp: added Gsplat feature to {Path.GetFileName(path)}");
            }
        }

        static void SetBool(SerializedObject so, string property, bool value)
        {
            var p = so.FindProperty(property);
            if (p != null) p.boolValue = value;
        }

        // ------------------------------------------------------------------
        // Scene: the layout from UNITY_SETUP.md section 8.
        // ------------------------------------------------------------------
        static void BuildScene(StringBuilder log)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var originGo = new GameObject("XR Origin");
            var origin = originGo.AddComponent<XROrigin>();
            var offset = new GameObject("Camera Offset");
            offset.transform.SetParent(originGo.transform, false);

            var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
            camGo.transform.SetParent(offset.transform, false);
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0f, 0f, 0f, 0f);   // alpha 0: passthrough shows behind
            cam.nearClipPlane = 0.05f;
            cam.farClipPlane = 100f;
            camGo.AddComponent<UniversalAdditionalCameraData>();
            camGo.AddComponent<ARCameraManager>();               // enabled = passthrough on

            var head = camGo.AddComponent<TrackedPoseDriver>();
            head.positionInput = Bind("Head Position", "<XRHMD>/centerEyePosition", "Vector3");
            head.rotationInput = Bind("Head Rotation", "<XRHMD>/centerEyeRotation", "Quaternion");
            head.trackingStateInput = Bind("Head Tracking State", "<XRHMD>/trackingState", "Integer");

            origin.Camera = cam;
            origin.CameraFloorOffsetObject = offset;
            origin.RequestedTrackingOriginMode = XROrigin.TrackingOriginMode.Floor;

            // Space Setup surfaces; SceneSurfaces enables this once USE_SCENE is granted.
            var planeManager = originGo.AddComponent<ARPlaneManager>();
            planeManager.requestedDetectionMode = UnityEngine.XR.ARSubsystems.PlaneDetectionMode.Horizontal;
            planeManager.enabled = false;
            new GameObject("AR Session").AddComponent<ARSession>();

            var splatRoot = new GameObject("SplatRoot");
            splatRoot.transform.position = SplatPosition;

            // Head-locked readout, low and to the left of the view.
            var hud = new GameObject("WristHud");
            hud.transform.SetParent(camGo.transform, false);
            hud.transform.localPosition = new Vector3(-0.15f, -0.12f, 0.6f);
            var text = hud.AddComponent<TextMesh>();
            var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.font = font;
            hud.GetComponent<MeshRenderer>().sharedMaterial = font.material;
            text.fontSize = 64;
            text.characterSize = 0.0025f;
            text.anchor = TextAnchor.UpperLeft;
            text.text = "loading...";

            var app = new GameObject("AppRoot");
            var browser = app.AddComponent<SplatSceneBrowser>();
            browser.splatRoot = splatRoot.transform;
            app.AddComponent<SplatPlacementRig>().splatRoot = splatRoot.transform;
            var perf = app.AddComponent<PerfHud>();
            perf.target = text;
            perf.browser = browser;

            var aim = BuildCaptureUi(offset.transform, app, browser, font);

            var surfaces = app.AddComponent<SceneSurfaces>();
            surfaces.planes = planeManager;
            var grab = app.AddComponent<SplatGrab>();
            grab.splatRoot = splatRoot.transform;
            grab.browser = browser;
            grab.hand = aim;
            grab.surfaces = surfaces;

            EditorSceneManager.SaveScene(scene, ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            log.AppendLine($"scene: {ScenePath} (XR Origin, SplatRoot at {SplatPosition}, AppRoot, HUD, " +
                           $"capture panel at {PanelPosition})");
        }

        /// <summary>The phone-capture loop's UI: see CaptureClient. Returns the
        /// right controller's aim pose, which SplatGrab shares.</summary>
        static Transform BuildCaptureUi(Transform cameraOffset, GameObject app, SplatSceneBrowser browser, Font font)
        {
            // Materials are assets so their URP shader is included in the build.
            var panelMat = UnlitMaterial("CapturePanel", Color.white);
            var rayMat = UnlitMaterial("CaptureRay", new Color(1f, 0.55f, 0.2f));
            var markMat = UnlitMaterial("CaptureMarker", new Color(0.2f, 1f, 0.4f));

            // OpenXR's pointer pose is the controller's aim ray, not its grip.
            var aimGo = new GameObject("Right Aim");
            aimGo.transform.SetParent(cameraOffset, false);
            var aim = aimGo.AddComponent<TrackedPoseDriver>();
            aim.positionInput = Bind("Aim Position", "<XRController>{RightHand}/pointerPosition", "Vector3");
            aim.rotationInput = Bind("Aim Rotation", "<XRController>{RightHand}/pointerRotation", "Quaternion");
            aim.trackingStateInput = Bind("Aim Tracking State", "<XRController>{RightHand}/trackingState", "Integer");
            var ray = aimGo.AddComponent<LineRenderer>();
            SetUpLine(ray, rayMat, 0.003f, 2, loop: false);

            var panel = GameObject.CreatePrimitive(PrimitiveType.Quad);   // faces -Z, toward the user
            panel.name = "CapturePanel";
            UnityEngine.Object.DestroyImmediate(panel.GetComponent<Collider>());
            panel.transform.position = PanelPosition;
            panel.GetComponent<MeshRenderer>().sharedMaterial = panelMat;

            var statusGo = new GameObject("CaptureStatus");
            statusGo.transform.position = PanelPosition + new Vector3(0f, 0.38f, 0f);
            var status = statusGo.AddComponent<TextMesh>();
            status.font = font;
            statusGo.GetComponent<MeshRenderer>().sharedMaterial = font.material;
            status.fontSize = 64;
            status.characterSize = 0.004f;
            status.anchor = TextAnchor.LowerCenter;
            status.alignment = TextAlignment.Center;
            status.text = "";

            var point = GameObject.CreatePrimitive(PrimitiveType.Quad);
            point.name = "PickPoint";
            UnityEngine.Object.DestroyImmediate(point.GetComponent<Collider>());
            point.transform.localScale = Vector3.one * 0.02f;
            point.GetComponent<MeshRenderer>().sharedMaterial = markMat;
            point.SetActive(false);

            var boxGo = new GameObject("PickBox");
            var box = boxGo.AddComponent<LineRenderer>();
            SetUpLine(box, markMat, 0.004f, 4, loop: true);

            var client = app.AddComponent<CaptureClient>();
            client.browser = browser;
            client.panel = panel.transform;
            client.panelRenderer = panel.GetComponent<MeshRenderer>();
            client.status = status;
            client.aim = aimGo.transform;
            client.ray = ray;
            client.pointMarker = point.transform;
            client.boxMarker = box;
            return aimGo.transform;
        }

        static InputActionProperty Bind(string name, string binding, string controlType) =>
            new InputActionProperty(new InputAction(name, binding: binding, expectedControlType: controlType));

        static Material UnlitMaterial(string name, Color color)
        {
            string path = $"{MaterialDir}/{name}.mat";
            var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (mat == null)
            {
                Directory.CreateDirectory(MaterialDir);
                mat = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
                AssetDatabase.CreateAsset(mat, path);
            }
            mat.SetColor("_BaseColor", color);
            EditorUtility.SetDirty(mat);
            return mat;
        }

        static void SetUpLine(LineRenderer line, Material mat, float width, int points, bool loop)
        {
            line.sharedMaterial = mat;
            line.widthMultiplier = width;
            line.positionCount = points;
            line.loop = loop;
            line.useWorldSpace = true;
            line.shadowCastingMode = ShadowCastingMode.Off;
            line.receiveShadows = false;
            line.enabled = false;
        }
    }
}
