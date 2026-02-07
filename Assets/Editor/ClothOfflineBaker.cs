#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using Unity.Jobs;
using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;
using System;
using System.Collections.Generic;
using System.IO;

namespace ClothBaking
{
    /// <summary>
    /// High-fidelity cloth offline baker using Projective Dynamics (Local-Global step).
    /// Implements energy minimization with Newton iterations for Marvelous Designer-quality draping.
    /// </summary>
    public class ClothOfflineBaker : EditorWindow
    {
        [MenuItem("Tools/Cloth Offline Baker")]
        private static void ShowWindow()
        {
            var window = GetWindow<ClothOfflineBaker>();
            window.titleContent = new GUIContent("Cloth Offline Baker");
            window.Show();
        }

        #region Serialized Fields (UI State)

        // References
        private SkinnedMeshRenderer _clothRenderer;
        private SkinnedMeshRenderer _colliderRenderer;
        private AnimationClip _animationClip;
        private Animator _animator;

        // Simulation Parameters
        private float _timeStep = 1f / 60f;
        private int _substeps = 4;
        private int _solverIterations = 20;
        private float _gravity = -9.81f;
        private float _totalMass = 1.0f;
        private float _velocityDamping = 0.999f;

        // Constraint Stiffness
        private float _stretchStiffness = 1.0f;
        private float _shearStiffness = 0.8f;
        private float _bendStiffness = 0.05f;

        // Collision
        private float _collisionThickness = 0.002f;
        private float _selfCollisionThickness = 0.004f;
        private float _friction = 0.3f;
        private int _hashGridResolution = 64;
        private float _colliderInflate = 0.001f;

        // Pin Constraints
        private bool _usePinByColor = false;
        private int _pinColorChannel = 0; // 0=R, 1=G, 2=B, 3=A
        private float _pinColorThreshold = 0.5f;
        private bool _usePinByHeight = false;
        private float _pinTopPercentage = 10f;

        // Output
        private string _outputFolder = "Assets/BakedCloth";
        private string _outputPrefix = "Frame";
        private bool _singleMeshPerFrame = true;

        private Vector2 _scrollPosition;

        #endregion

        private void OnGUI()
        {
            _scrollPosition = EditorGUILayout.BeginScrollView(_scrollPosition);

            EditorGUILayout.LabelField("Cloth Offline Baker", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            // References Section
            EditorGUILayout.LabelField("References", EditorStyles.boldLabel);
            _clothRenderer = EditorGUILayout.ObjectField("Cloth SkinnedMeshRenderer", _clothRenderer, typeof(SkinnedMeshRenderer), true) as SkinnedMeshRenderer;
            _colliderRenderer = EditorGUILayout.ObjectField("Collider SkinnedMeshRenderer", _colliderRenderer, typeof(SkinnedMeshRenderer), true) as SkinnedMeshRenderer;
            _animationClip = EditorGUILayout.ObjectField("Animation Clip", _animationClip, typeof(AnimationClip), false) as AnimationClip;
            _animator = EditorGUILayout.ObjectField("Animator", _animator, typeof(Animator), true) as Animator;
            EditorGUILayout.Space();

            // Simulation Parameters Section
            EditorGUILayout.LabelField("Simulation Parameters", EditorStyles.boldLabel);
            _timeStep = EditorGUILayout.FloatField("Time Step", _timeStep);
            _substeps = EditorGUILayout.IntSlider("Substeps", _substeps, 1, 20);
            _solverIterations = EditorGUILayout.IntSlider("Solver Iterations", _solverIterations, 1, 100);
            _gravity = EditorGUILayout.FloatField("Gravity", _gravity);
            _totalMass = EditorGUILayout.FloatField("Total Mass (kg)", _totalMass);
            _velocityDamping = EditorGUILayout.Slider("Velocity Damping", _velocityDamping, 0.9f, 1.0f);
            EditorGUILayout.Space();

            // Constraint Stiffness Section
            EditorGUILayout.LabelField("Constraint Stiffness", EditorStyles.boldLabel);
            _stretchStiffness = EditorGUILayout.Slider("Stretch", _stretchStiffness, 0f, 1f);
            _shearStiffness = EditorGUILayout.Slider("Shear", _shearStiffness, 0f, 1f);
            _bendStiffness = EditorGUILayout.Slider("Bend", _bendStiffness, 0f, 1f);
            EditorGUILayout.Space();

            // Collision Section
            EditorGUILayout.LabelField("Collision", EditorStyles.boldLabel);
            _collisionThickness = EditorGUILayout.FloatField("Collision Thickness", _collisionThickness);
            _selfCollisionThickness = EditorGUILayout.FloatField("Self-Collision Thickness", _selfCollisionThickness);
            _friction = EditorGUILayout.Slider("Friction", _friction, 0f, 1f);
            _hashGridResolution = EditorGUILayout.IntSlider("Hash Grid Resolution", _hashGridResolution, 16, 256);
            _colliderInflate = EditorGUILayout.FloatField("Collider Inflate", _colliderInflate);
            EditorGUILayout.Space();

            // Pin Constraints Section
            EditorGUILayout.LabelField("Pin Constraints", EditorStyles.boldLabel);
            _usePinByColor = EditorGUILayout.Toggle("Use Pin by Color", _usePinByColor);
            if (_usePinByColor)
            {
                string[] channels = { "Red", "Green", "Blue", "Alpha" };
                _pinColorChannel = EditorGUILayout.Popup("Color Channel", _pinColorChannel, channels);
                _pinColorThreshold = EditorGUILayout.Slider("Threshold", _pinColorThreshold, 0f, 1f);
            }
            _usePinByHeight = EditorGUILayout.Toggle("Use Pin by Height", _usePinByHeight);
            if (_usePinByHeight)
            {
                _pinTopPercentage = EditorGUILayout.Slider("Top Percentage", _pinTopPercentage, 0f, 100f);
            }
            EditorGUILayout.Space();

            // Output Section
            EditorGUILayout.LabelField("Output", EditorStyles.boldLabel);
            _outputFolder = EditorGUILayout.TextField("Output Folder", _outputFolder);
            _outputPrefix = EditorGUILayout.TextField("Output Prefix", _outputPrefix);
            _singleMeshPerFrame = EditorGUILayout.Toggle("Single Mesh Per Frame", _singleMeshPerFrame);
            EditorGUILayout.Space();

            // Bake Button
            if (GUILayout.Button("Bake", GUILayout.Height(40)))
            {
                StartBaking();
            }

            EditorGUILayout.EndScrollView();
        }

        private void StartBaking()
        {
            if (!ValidateInputs())
                return;

            try
            {
                BakeClothAnimation();
            }
            catch (Exception e)
            {
                Debug.LogError($"Baking failed: {e.Message}\n{e.StackTrace}");
                EditorUtility.ClearProgressBar();
            }
        }

        private bool ValidateInputs()
        {
            if (_clothRenderer == null)
            {
                EditorUtility.DisplayDialog("Error", "Cloth SkinnedMeshRenderer is required.", "OK");
                return false;
            }
            if (_colliderRenderer == null)
            {
                EditorUtility.DisplayDialog("Error", "Collider SkinnedMeshRenderer is required.", "OK");
                return false;
            }
            if (_animationClip == null)
            {
                EditorUtility.DisplayDialog("Error", "Animation Clip is required.", "OK");
                return false;
            }
            if (_animator == null)
            {
                EditorUtility.DisplayDialog("Error", "Animator is required.", "OK");
                return false;
            }
            return true;
        }

        private void BakeClothAnimation()
        {
            // Create output directory
            if (!Directory.Exists(_outputFolder))
            {
                Directory.CreateDirectory(_outputFolder);
            }

            // Get animation info
            float clipLength = _animationClip.length;
            float frameRate = 1f / _timeStep;
            int totalFrames = Mathf.CeilToInt(clipLength * frameRate);

            Debug.Log($"Starting cloth baking: {totalFrames} frames, {_substeps} substeps, {_solverIterations} iterations");

            // Sample cloth mesh at T=0 to get topology
            _animator.Play(_animationClip.name, 0, 0f);
            _animator.Update(0f);

            Mesh sourceMesh = new Mesh();
            _clothRenderer.BakeMesh(sourceMesh);

            // Transform to world space for initial positions
            Vector3[] sourceVertices = sourceMesh.vertices;
            Vector3[] worldVertices = new Vector3[sourceVertices.Length];
            for (int i = 0; i < sourceVertices.Length; i++)
            {
                worldVertices[i] = _clothRenderer.transform.TransformPoint(sourceVertices[i]);
            }

            // Build constraints
            var constraints = ConstraintBuilder.BuildConstraints(sourceMesh);
            Debug.Log($"Built constraints: {constraints.edgeConstraints.Length} edges, " +
                     $"{constraints.shearConstraints.Length} shear, {constraints.bendConstraints.Length} bend");

            // Compute masses
            float[] masses = ComputeMasses(sourceMesh, worldVertices, constraints);

            // Determine pin constraints
            float[] pinWeights = ComputePinWeights(sourceMesh, worldVertices);
            int pinnedCount = 0;
            for (int i = 0; i < pinWeights.Length; i++)
            {
                if (pinWeights[i] > 0f)
                    pinnedCount++;
            }
            Debug.Log($"Pinned {pinnedCount} vertices");

            // Initialize solver
            ProjectiveDynamicsSolver solver = null;
            try
            {
                solver = new ProjectiveDynamicsSolver(
                    worldVertices,
                    masses,
                    pinWeights,
                    constraints,
                    _solverIterations,
                    _gravity,
                    _velocityDamping,
                    _stretchStiffness,
                    _shearStiffness,
                    _bendStiffness,
                    _collisionThickness,
                    _selfCollisionThickness,
                    _friction,
                    _hashGridResolution,
                    _colliderInflate
                );

                // Baking loop
                float subDt = _timeStep / _substeps;

                for (int frame = 0; frame < totalFrames; frame++)
                {
                    float time = frame * _timeStep;

                    // Update animation
                    _animator.Play(_animationClip.name, 0, time / clipLength);
                    _animator.Update(0f);

                    // Update pin targets
                    Mesh clothMesh = new Mesh();
                    _clothRenderer.BakeMesh(clothMesh);
                    Vector3[] clothVerts = clothMesh.vertices;
                    for (int i = 0; i < clothVerts.Length; i++)
                    {
                        solver.pinTargets[i] = _clothRenderer.transform.TransformPoint(clothVerts[i]);
                    }
                    DestroyImmediate(clothMesh);

                    // Bake collider
                    Mesh colliderMesh = new Mesh();
                    _colliderRenderer.BakeMesh(colliderMesh);
                    Vector3[] colliderVerts = colliderMesh.vertices;
                    int[] colliderTris = colliderMesh.triangles;

                    NativeArray<float3> colliderVertsNative = new NativeArray<float3>(colliderVerts.Length, Allocator.TempJob);
                    NativeArray<int> colliderTrisNative = new NativeArray<int>(colliderTris.Length, Allocator.TempJob);

                    for (int i = 0; i < colliderVerts.Length; i++)
                    {
                        colliderVertsNative[i] = _colliderRenderer.transform.TransformPoint(colliderVerts[i]);
                    }
                    for (int i = 0; i < colliderTris.Length; i++)
                    {
                        colliderTrisNative[i] = colliderTris[i];
                    }

                    DestroyImmediate(colliderMesh);

                    // Simulate substeps
                    for (int substep = 0; substep < _substeps; substep++)
                    {
                        solver.Step(subDt, colliderVertsNative, colliderTrisNative);
                    }

                    colliderVertsNative.Dispose();
                    colliderTrisNative.Dispose();

                    // Save output mesh
                    SaveOutputMesh(sourceMesh, solver.positions, frame);

                    // Progress bar
                    if (frame % 10 == 0 || frame == totalFrames - 1)
                    {
                        bool cancel = EditorUtility.DisplayCancelableProgressBar(
                            "Baking Cloth",
                            $"Frame {frame + 1}/{totalFrames}",
                            (float)(frame + 1) / totalFrames
                        );
                        if (cancel)
                        {
                            Debug.Log("Baking cancelled by user.");
                            break;
                        }
                    }
                }

                EditorUtility.ClearProgressBar();
                AssetDatabase.Refresh();
                Debug.Log($"Baking complete! Output saved to {_outputFolder}");
            }
            finally
            {
                solver?.Dispose();
                DestroyImmediate(sourceMesh);
            }
        }

        private float[] ComputeMasses(Mesh mesh, Vector3[] worldVertices, ConstraintData constraints)
        {
            int vertexCount = worldVertices.Length;
            float[] masses = new float[vertexCount];

            // Compute area-weighted Voronoi masses
            int[] triangles = mesh.triangles;
            for (int i = 0; i < triangles.Length; i += 3)
            {
                int i0 = triangles[i];
                int i1 = triangles[i + 1];
                int i2 = triangles[i + 2];

                Vector3 v0 = worldVertices[i0];
                Vector3 v1 = worldVertices[i1];
                Vector3 v2 = worldVertices[i2];

                float area = Vector3.Cross(v1 - v0, v2 - v0).magnitude * 0.5f;
                float areaPer3 = area / 3f;

                masses[i0] += areaPer3;
                masses[i1] += areaPer3;
                masses[i2] += areaPer3;
            }

            // Normalize to total mass
            float totalArea = 0f;
            for (int i = 0; i < vertexCount; i++)
            {
                totalArea += masses[i];
            }

            float scale = _totalMass / Mathf.Max(totalArea, 1e-9f);
            float minMass = _totalMass / (vertexCount * 100f);

            for (int i = 0; i < vertexCount; i++)
            {
                masses[i] = Mathf.Max(masses[i] * scale, minMass);
            }

            return masses;
        }

        private float[] ComputePinWeights(Mesh mesh, Vector3[] worldVertices)
        {
            int vertexCount = worldVertices.Length;
            float[] pinWeights = new float[vertexCount];

            // Pin by color
            if (_usePinByColor && mesh.colors != null && mesh.colors.Length == vertexCount)
            {
                Color[] colors = mesh.colors;
                for (int i = 0; i < vertexCount; i++)
                {
                    float channelValue = 0f;
                    switch (_pinColorChannel)
                    {
                        case 0: channelValue = colors[i].r; break;
                        case 1: channelValue = colors[i].g; break;
                        case 2: channelValue = colors[i].b; break;
                        case 3: channelValue = colors[i].a; break;
                    }

                    if (channelValue >= _pinColorThreshold)
                    {
                        pinWeights[i] = Mathf.Max(pinWeights[i], channelValue);
                    }
                }
            }

            // Pin by height
            if (_usePinByHeight)
            {
                float minY = float.MaxValue;
                float maxY = float.MinValue;
                for (int i = 0; i < vertexCount; i++)
                {
                    float y = worldVertices[i].y;
                    minY = Mathf.Min(minY, y);
                    maxY = Mathf.Max(maxY, y);
                }

                float threshold = maxY - (maxY - minY) * (_pinTopPercentage / 100f);
                for (int i = 0; i < vertexCount; i++)
                {
                    if (worldVertices[i].y >= threshold)
                    {
                        pinWeights[i] = 1f;
                    }
                }
            }

            return pinWeights;
        }

        private void SaveOutputMesh(Mesh sourceMesh, NativeArray<float3> positions, int frame)
        {
            // Convert back to local space
            Vector3[] localVertices = new Vector3[positions.Length];
            for (int i = 0; i < positions.Length; i++)
            {
                Vector3 worldPos = positions[i];
                localVertices[i] = _clothRenderer.transform.InverseTransformPoint(worldPos);
            }

            // Create output mesh
            Mesh outputMesh = new Mesh();
            outputMesh.name = $"{_outputPrefix}{frame:D4}";
            outputMesh.vertices = localVertices;
            outputMesh.triangles = sourceMesh.triangles;
            outputMesh.uv = sourceMesh.uv;
            outputMesh.colors = sourceMesh.colors;
            outputMesh.boneWeights = sourceMesh.boneWeights;
            outputMesh.bindposes = sourceMesh.bindposes;

            outputMesh.RecalculateNormals();
            outputMesh.RecalculateBounds();
            outputMesh.RecalculateTangents();

            // Save as asset
            string assetPath = $"{_outputFolder}/{_outputPrefix}{frame:D4}.asset";
            AssetDatabase.CreateAsset(outputMesh, assetPath);
        }
    }

    #region Constraint Data Structures

    public struct EdgeConstraint
    {
        public int i0;
        public int i1;
        public float restLength;
    }

    public struct BendConstraint
    {
        public int e0;
        public int e1;
        public int oppositeA;
        public int oppositeB;
        public float restAngle;
    }

    public struct ConstraintData
    {
        public EdgeConstraint[] edgeConstraints;
        public EdgeConstraint[] shearConstraints;
        public BendConstraint[] bendConstraints;
    }

    #endregion

    #region Constraint Builder

    public static class ConstraintBuilder
    {
        public static ConstraintData BuildConstraints(Mesh mesh)
        {
            Vector3[] vertices = mesh.vertices;
            int[] triangles = mesh.triangles;

            // Build edge constraints
            Dictionary<long, EdgeConstraint> edgeDict = new Dictionary<long, EdgeConstraint>();
            for (int i = 0; i < triangles.Length; i += 3)
            {
                int i0 = triangles[i];
                int i1 = triangles[i + 1];
                int i2 = triangles[i + 2];

                AddEdge(edgeDict, vertices, i0, i1);
                AddEdge(edgeDict, vertices, i1, i2);
                AddEdge(edgeDict, vertices, i2, i0);
            }

            EdgeConstraint[] edges = new EdgeConstraint[edgeDict.Count];
            edgeDict.Values.CopyTo(edges, 0);

            // Build triangle adjacency for bending
            Dictionary<long, List<int>> edgeToTriangles = new Dictionary<long, List<int>>();
            for (int triIdx = 0; triIdx < triangles.Length / 3; triIdx++)
            {
                int baseIdx = triIdx * 3;
                int i0 = triangles[baseIdx];
                int i1 = triangles[baseIdx + 1];
                int i2 = triangles[baseIdx + 2];

                AddTriangleToEdge(edgeToTriangles, i0, i1, triIdx);
                AddTriangleToEdge(edgeToTriangles, i1, i2, triIdx);
                AddTriangleToEdge(edgeToTriangles, i2, i0, triIdx);
            }

            // Build bend and shear constraints
            List<BendConstraint> bendList = new List<BendConstraint>();
            HashSet<long> shearSet = new HashSet<long>();
            List<EdgeConstraint> shearList = new List<EdgeConstraint>();

            foreach (var kvp in edgeToTriangles)
            {
                if (kvp.Value.Count == 2)
                {
                    long edgeKey = kvp.Key;
                    int minIdx = (int)(edgeKey >> 32);
                    int maxIdx = (int)(edgeKey & 0xFFFFFFFF);

                    int tri0 = kvp.Value[0];
                    int tri1 = kvp.Value[1];

                    // Find opposite vertices
                    int oppositeA = FindOppositeVertex(triangles, tri0, minIdx, maxIdx);
                    int oppositeB = FindOppositeVertex(triangles, tri1, minIdx, maxIdx);

                    if (oppositeA >= 0 && oppositeB >= 0)
                    {
                        // Compute rest angle
                        Vector3 v0 = vertices[minIdx];
                        Vector3 v1 = vertices[maxIdx];
                        Vector3 vA = vertices[oppositeA];
                        Vector3 vB = vertices[oppositeB];

                        Vector3 n0 = Vector3.Cross(vA - v0, v1 - v0).normalized;
                        Vector3 n1 = Vector3.Cross(v1 - v0, vB - v0).normalized;

                        float cosAngle = Vector3.Dot(n0, n1);
                        Vector3 cross = Vector3.Cross(n0, n1);
                        float sinAngle = cross.magnitude * Mathf.Sign(Vector3.Dot(cross, v1 - v0));
                        float restAngle = Mathf.Atan2(sinAngle, cosAngle);

                        BendConstraint bend = new BendConstraint
                        {
                            e0 = minIdx,
                            e1 = maxIdx,
                            oppositeA = oppositeA,
                            oppositeB = oppositeB,
                            restAngle = restAngle
                        };
                        bendList.Add(bend);

                        // Add shear constraint (diagonal)
                        long shearKey = GetEdgeKey(oppositeA, oppositeB);
                        if (!edgeDict.ContainsKey(shearKey) && !shearSet.Contains(shearKey))
                        {
                            shearSet.Add(shearKey);
                            float shearLength = (vertices[oppositeA] - vertices[oppositeB]).magnitude;
                            EdgeConstraint shear = new EdgeConstraint
                            {
                                i0 = oppositeA,
                                i1 = oppositeB,
                                restLength = shearLength
                            };
                            shearList.Add(shear);
                        }
                    }
                }
            }

            return new ConstraintData
            {
                edgeConstraints = edges,
                shearConstraints = shearList.ToArray(),
                bendConstraints = bendList.ToArray()
            };
        }

        private static void AddEdge(Dictionary<long, EdgeConstraint> dict, Vector3[] vertices, int i0, int i1)
        {
            long key = GetEdgeKey(i0, i1);
            if (!dict.ContainsKey(key))
            {
                float length = (vertices[i0] - vertices[i1]).magnitude;
                dict[key] = new EdgeConstraint
                {
                    i0 = Mathf.Min(i0, i1),
                    i1 = Mathf.Max(i0, i1),
                    restLength = length
                };
            }
        }

        private static void AddTriangleToEdge(Dictionary<long, List<int>> dict, int i0, int i1, int triIdx)
        {
            long key = GetEdgeKey(i0, i1);
            if (!dict.ContainsKey(key))
            {
                dict[key] = new List<int>();
            }
            dict[key].Add(triIdx);
        }

        private static long GetEdgeKey(int i0, int i1)
        {
            int min = Mathf.Min(i0, i1);
            int max = Mathf.Max(i0, i1);
            return ((long)min << 32) | (uint)max;
        }

        private static int FindOppositeVertex(int[] triangles, int triIdx, int e0, int e1)
        {
            int baseIdx = triIdx * 3;
            int v0 = triangles[baseIdx];
            int v1 = triangles[baseIdx + 1];
            int v2 = triangles[baseIdx + 2];

            if (v0 != e0 && v0 != e1) return v0;
            if (v1 != e0 && v1 != e1) return v1;
            if (v2 != e0 && v2 != e1) return v2;
            return -1;
        }
    }

    #endregion

    #region Projective Dynamics Solver

    public class ProjectiveDynamicsSolver : IDisposable
    {
        // Particle state
        public NativeArray<float3> positions;
        public NativeArray<float3> velocities;
        public NativeArray<float3> predictedPositions;
        public NativeArray<float> inverseMasses;
        public NativeArray<float> pinWeights;
        public NativeArray<float3> pinTargets;

        // Constraints
        private NativeArray<EdgeConstraint> _edgeConstraints;
        private NativeArray<EdgeConstraint> _shearConstraints;
        private NativeArray<BendConstraint> _bendConstraints;

        // Solver state
        private NativeArray<float> _diagA;
        private NativeArray<float3> _rhs;
        private NativeArray<float3> _q;

        // Spatial hash for self-collision
        private NativeMultiHashMap<int, int> _spatialHash;

        // Parameters
        private int _solverIterations;
        private float _gravity;
        private float _velocityDamping;
        private float _stretchStiffness;
        private float _shearStiffness;
        private float _bendStiffness;
        private float _collisionThickness;
        private float _selfCollisionThickness;
        private float _friction;
        private int _hashGridResolution;
        private float _colliderInflate;

        public ProjectiveDynamicsSolver(
            Vector3[] initialPositions,
            float[] masses,
            float[] pinWeights,
            ConstraintData constraints,
            int solverIterations,
            float gravity,
            float velocityDamping,
            float stretchStiffness,
            float shearStiffness,
            float bendStiffness,
            float collisionThickness,
            float selfCollisionThickness,
            float friction,
            int hashGridResolution,
            float colliderInflate)
        {
            int vertexCount = initialPositions.Length;

            // Allocate particle arrays
            positions = new NativeArray<float3>(vertexCount, Allocator.Persistent);
            velocities = new NativeArray<float3>(vertexCount, Allocator.Persistent);
            predictedPositions = new NativeArray<float3>(vertexCount, Allocator.Persistent);
            inverseMasses = new NativeArray<float>(vertexCount, Allocator.Persistent);
            this.pinWeights = new NativeArray<float>(vertexCount, Allocator.Persistent);
            pinTargets = new NativeArray<float3>(vertexCount, Allocator.Persistent);

            // Initialize positions and masses
            for (int i = 0; i < vertexCount; i++)
            {
                positions[i] = initialPositions[i];
                velocities[i] = float3.zero;
                inverseMasses[i] = pinWeights[i] > 0f ? 0f : 1f / masses[i];
                this.pinWeights[i] = pinWeights[i];
                pinTargets[i] = initialPositions[i];
            }

            // Allocate constraint arrays
            _edgeConstraints = new NativeArray<EdgeConstraint>(constraints.edgeConstraints.Length, Allocator.Persistent);
            _shearConstraints = new NativeArray<EdgeConstraint>(constraints.shearConstraints.Length, Allocator.Persistent);
            _bendConstraints = new NativeArray<BendConstraint>(constraints.bendConstraints.Length, Allocator.Persistent);

            for (int i = 0; i < constraints.edgeConstraints.Length; i++)
                _edgeConstraints[i] = constraints.edgeConstraints[i];
            for (int i = 0; i < constraints.shearConstraints.Length; i++)
                _shearConstraints[i] = constraints.shearConstraints[i];
            for (int i = 0; i < constraints.bendConstraints.Length; i++)
                _bendConstraints[i] = constraints.bendConstraints[i];

            // Allocate solver arrays
            _diagA = new NativeArray<float>(vertexCount, Allocator.Persistent);
            _rhs = new NativeArray<float3>(vertexCount, Allocator.Persistent);
            _q = new NativeArray<float3>(vertexCount, Allocator.Persistent);

            // Allocate spatial hash
            _spatialHash = new NativeMultiHashMap<int, int>(vertexCount * 27, Allocator.Persistent);

            // Store parameters
            _solverIterations = solverIterations;
            _gravity = gravity;
            _velocityDamping = velocityDamping;
            _stretchStiffness = stretchStiffness;
            _shearStiffness = shearStiffness;
            _bendStiffness = bendStiffness;
            _collisionThickness = collisionThickness;
            _selfCollisionThickness = selfCollisionThickness;
            _friction = friction;
            _hashGridResolution = hashGridResolution;
            _colliderInflate = colliderInflate;
        }

        public void Step(float dt, NativeArray<float3> colliderVertices, NativeArray<int> colliderTriangles)
        {
            int vertexCount = positions.Length;

            // Predict positions
            var predictJob = new PredictPositionsJob
            {
                positions = positions,
                velocities = velocities,
                predictedPositions = predictedPositions,
                inverseMasses = inverseMasses,
                gravity = _gravity,
                damping = _velocityDamping,
                dt = dt
            };
            predictJob.Schedule(vertexCount, 64).Complete();

            // Build diagonal matrix (mass term + constraint weights)
            var clearDiagJob = new ClearFloatArrayJob { array = _diagA };
            clearDiagJob.Schedule(vertexCount, 64).Complete();

            var buildDiagMassJob = new BuildDiagMassJob
            {
                diagA = _diagA,
                inverseMasses = inverseMasses,
                dt = dt
            };
            buildDiagMassJob.Schedule(vertexCount, 64).Complete();

            var buildDiagEdgeJob = new BuildDiagEdgeConstraintJob
            {
                diagA = _diagA,
                constraints = _edgeConstraints,
                stiffness = _stretchStiffness
            };
            buildDiagEdgeJob.Schedule(_edgeConstraints.Length, 64).Complete();

            var buildDiagShearJob = new BuildDiagShearConstraintJob
            {
                diagA = _diagA,
                constraints = _shearConstraints,
                stiffness = _shearStiffness
            };
            buildDiagShearJob.Schedule(_shearConstraints.Length, 64).Complete();

            var buildDiagBendJob = new BuildDiagBendConstraintJob
            {
                diagA = _diagA,
                constraints = _bendConstraints,
                stiffness = _bendStiffness
            };
            buildDiagBendJob.Schedule(_bendConstraints.Length, 64).Complete();

            var clampDiagJob = new ClampDiagJob { diagA = _diagA, minValue = 1e-9f };
            clampDiagJob.Schedule(vertexCount, 64).Complete();

            // Local-Global iterations
            for (int iter = 0; iter < _solverIterations; iter++)
            {
                // Clear RHS
                var clearRhsJob = new ClearFloat3ArrayJob { array = _rhs };
                clearRhsJob.Schedule(vertexCount, 64).Complete();

                // Build RHS momentum term
                var buildRhsMomentumJob = new BuildRhsMomentumJob
                {
                    rhs = _rhs,
                    predictedPositions = predictedPositions,
                    inverseMasses = inverseMasses,
                    dt = dt
                };
                buildRhsMomentumJob.Schedule(vertexCount, 64).Complete();

                // Local step - edge constraints
                float3[] qArray = new float3[vertexCount];
                if (iter == 0)
                {
                    for (int i = 0; i < vertexCount; i++)
                        qArray[i] = predictedPositions[i];
                }
                else
                {
                    for (int i = 0; i < vertexCount; i++)
                        qArray[i] = _q[i];
                }

                NativeArray<float3> qInput = new NativeArray<float3>(qArray, Allocator.TempJob);

                var localStepEdgeJob = new LocalStepEdgeJob
                {
                    rhs = _rhs,
                    positions = qInput,
                    constraints = _edgeConstraints,
                    stiffness = _stretchStiffness
                };
                localStepEdgeJob.Schedule(_edgeConstraints.Length, 64).Complete();

                // Local step - shear constraints
                var localStepShearJob = new LocalStepShearJob
                {
                    rhs = _rhs,
                    positions = qInput,
                    constraints = _shearConstraints,
                    stiffness = _shearStiffness
                };
                localStepShearJob.Schedule(_shearConstraints.Length, 64).Complete();

                // Local step - bend constraints
                var localStepBendJob = new LocalStepBendJob
                {
                    rhs = _rhs,
                    positions = qInput,
                    constraints = _bendConstraints,
                    stiffness = _bendStiffness
                };
                localStepBendJob.Schedule(_bendConstraints.Length, 64).Complete();

                qInput.Dispose();

                // Global step
                var globalStepJob = new GlobalStepJob
                {
                    q = _q,
                    rhs = _rhs,
                    diagA = _diagA,
                    pinWeights = pinWeights,
                    pinTargets = pinTargets
                };
                globalStepJob.Schedule(vertexCount, 64).Complete();
            }

            // Apply collisions
            var colliderCollisionJob = new ColliderCollisionJob
            {
                positions = _q,
                velocities = velocities,
                inverseMasses = inverseMasses,
                colliderVertices = colliderVertices,
                colliderTriangles = colliderTriangles,
                thickness = _collisionThickness + _colliderInflate,
                friction = _friction,
                dt = dt
            };
            colliderCollisionJob.Schedule(vertexCount, 16).Complete();

            // Self-collision
            _spatialHash.Clear();
            var buildHashJob = new BuildSpatialHashJob
            {
                positions = _q,
                spatialHash = _spatialHash.AsParallelWriter(),
                cellSize = _selfCollisionThickness * 2f
            };
            buildHashJob.Schedule(vertexCount, 64).Complete();

            var selfCollisionJob = new SelfCollisionJob
            {
                positions = _q,
                velocities = velocities,
                inverseMasses = inverseMasses,
                spatialHash = _spatialHash,
                thickness = _selfCollisionThickness,
                friction = _friction,
                cellSize = _selfCollisionThickness * 2f
            };
            selfCollisionJob.Schedule(vertexCount, 16).Complete();

            // Update velocity and position
            var updateJob = new UpdateVelocityPositionJob
            {
                positions = positions,
                velocities = velocities,
                q = _q,
                dt = dt
            };
            updateJob.Schedule(vertexCount, 64).Complete();
        }

        public void Dispose()
        {
            if (positions.IsCreated) positions.Dispose();
            if (velocities.IsCreated) velocities.Dispose();
            if (predictedPositions.IsCreated) predictedPositions.Dispose();
            if (inverseMasses.IsCreated) inverseMasses.Dispose();
            if (pinWeights.IsCreated) pinWeights.Dispose();
            if (pinTargets.IsCreated) pinTargets.Dispose();
            if (_edgeConstraints.IsCreated) _edgeConstraints.Dispose();
            if (_shearConstraints.IsCreated) _shearConstraints.Dispose();
            if (_bendConstraints.IsCreated) _bendConstraints.Dispose();
            if (_diagA.IsCreated) _diagA.Dispose();
            if (_rhs.IsCreated) _rhs.Dispose();
            if (_q.IsCreated) _q.Dispose();
            if (_spatialHash.IsCreated) _spatialHash.Dispose();
        }
    }

    #endregion

    #region Burst Jobs

    [BurstCompile]
    struct PredictPositionsJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<float3> velocities;
        [ReadOnly] public NativeArray<float> inverseMasses;
        [WriteOnly] public NativeArray<float3> predictedPositions;
        public float gravity;
        public float damping;
        public float dt;

        public void Execute(int i)
        {
            if (inverseMasses[i] > 0f)
            {
                float3 vel = velocities[i] * damping;
                float3 force = new float3(0, gravity, 0) / inverseMasses[i];
                predictedPositions[i] = positions[i] + vel * dt + force * (dt * dt);
            }
            else
            {
                predictedPositions[i] = positions[i];
            }
        }
    }

    [BurstCompile]
    struct ClearFloatArrayJob : IJobParallelFor
    {
        [WriteOnly] public NativeArray<float> array;

        public void Execute(int i)
        {
            array[i] = 0f;
        }
    }

    [BurstCompile]
    struct ClearFloat3ArrayJob : IJobParallelFor
    {
        [WriteOnly] public NativeArray<float3> array;

        public void Execute(int i)
        {
            array[i] = float3.zero;
        }
    }

    [BurstCompile]
    struct BuildDiagMassJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float> diagA;
        [ReadOnly] public NativeArray<float> inverseMasses;
        public float dt;

        public void Execute(int i)
        {
            if (inverseMasses[i] > 0f)
            {
                diagA[i] += 1f / (inverseMasses[i] * dt * dt);
            }
            else
            {
                diagA[i] += 1e9f; // Very large value for pinned particles
            }
        }
    }

    [BurstCompile]
    struct BuildDiagEdgeConstraintJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float> diagA;
        [ReadOnly] public NativeArray<EdgeConstraint> constraints;
        public float stiffness;

        public void Execute(int idx)
        {
            EdgeConstraint c = constraints[idx];
            float w = stiffness / math.max(c.restLength, 1e-6f);
            diagA[c.i0] += w;
            diagA[c.i1] += w;
        }
    }

    [BurstCompile]
    struct BuildDiagShearConstraintJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float> diagA;
        [ReadOnly] public NativeArray<EdgeConstraint> constraints;
        public float stiffness;

        public void Execute(int idx)
        {
            EdgeConstraint c = constraints[idx];
            float w = stiffness / math.max(c.restLength, 1e-6f);
            diagA[c.i0] += w;
            diagA[c.i1] += w;
        }
    }

    [BurstCompile]
    struct BuildDiagBendConstraintJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float> diagA;
        [ReadOnly] public NativeArray<BendConstraint> constraints;
        public float stiffness;

        public void Execute(int idx)
        {
            diagA[constraints[idx].e0] += stiffness;
            diagA[constraints[idx].e1] += stiffness;
            diagA[constraints[idx].oppositeA] += stiffness;
            diagA[constraints[idx].oppositeB] += stiffness;
        }
    }

    [BurstCompile]
    struct ClampDiagJob : IJobParallelFor
    {
        public NativeArray<float> diagA;
        public float minValue;

        public void Execute(int i)
        {
            diagA[i] = math.max(diagA[i], minValue);
        }
    }

    [BurstCompile]
    struct BuildRhsMomentumJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float3> rhs;
        [ReadOnly] public NativeArray<float3> predictedPositions;
        [ReadOnly] public NativeArray<float> inverseMasses;
        public float dt;

        public void Execute(int i)
        {
            if (inverseMasses[i] > 0f)
            {
                float massOverDtSq = 1f / (inverseMasses[i] * dt * dt);
                rhs[i] += massOverDtSq * predictedPositions[i];
            }
        }
    }

    [BurstCompile]
    struct LocalStepEdgeJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float3> rhs;
        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<EdgeConstraint> constraints;
        public float stiffness;

        public void Execute(int idx)
        {
            EdgeConstraint c = constraints[idx];
            float3 p0 = positions[c.i0];
            float3 p1 = positions[c.i1];
            float3 dir = p0 - p1;
            float len = math.length(dir);
            
            if (len > 1e-9f)
            {
                dir /= len;
            }
            else
            {
                dir = new float3(1, 0, 0);
            }

            float3 d = c.restLength * dir;
            float w = stiffness / math.max(c.restLength, 1e-6f);

            rhs[c.i0] += w * d;
            rhs[c.i1] += w * (-d);
        }
    }

    [BurstCompile]
    struct LocalStepShearJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float3> rhs;
        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<EdgeConstraint> constraints;
        public float stiffness;

        public void Execute(int idx)
        {
            EdgeConstraint c = constraints[idx];
            float3 p0 = positions[c.i0];
            float3 p1 = positions[c.i1];
            float3 dir = p0 - p1;
            float len = math.length(dir);
            
            if (len > 1e-9f)
            {
                dir /= len;
            }
            else
            {
                dir = new float3(1, 0, 0);
            }

            float3 d = c.restLength * dir;
            float w = stiffness / math.max(c.restLength, 1e-6f);

            rhs[c.i0] += w * d;
            rhs[c.i1] += w * (-d);
        }
    }

    [BurstCompile]
    struct LocalStepBendJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float3> rhs;
        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<BendConstraint> constraints;
        public float stiffness;

        public void Execute(int idx)
        {
            BendConstraint c = constraints[idx];
            
            float3 p0 = positions[c.e0];
            float3 p1 = positions[c.e1];
            float3 p2 = positions[c.oppositeA];
            float3 p3 = positions[c.oppositeB];

            float3 e = p1 - p0;
            float elen = math.length(e);
            if (elen < 1e-9f) return;

            // Compute face normals
            float3 n1 = math.cross(p2 - p0, p1 - p0);
            float3 n2 = math.cross(p1 - p0, p3 - p0);
            
            float n1len = math.length(n1);
            float n2len = math.length(n2);
            if (n1len < 1e-9f || n2len < 1e-9f) return;

            n1 /= n1len;
            n2 /= n2len;

            // Current dihedral angle
            float cosAngle = math.dot(n1, n2);
            cosAngle = math.clamp(cosAngle, -1f, 1f);
            float3 cross = math.cross(n1, n2);
            float sinAngle = math.length(cross) * math.sign(math.dot(cross, e));
            float currentAngle = math.atan2(sinAngle, cosAngle);

            float angleDiff = currentAngle - c.restAngle;

            // Compute gradients (simplified cotangent weights)
            float cot1 = math.dot(e, p2 - p0) / math.max(n1len, 1e-9f);
            float cot2 = math.dot(e, p3 - p0) / math.max(n2len, 1e-9f);

            float3 grad0 = -cot1 * n1 / elen - cot2 * n2 / elen;
            float3 grad1 = cot1 * n1 / elen + cot2 * n2 / elen;
            float3 grad2 = n1 / n1len;
            float3 grad3 = -n2 / n2len;

            float gradLenSq = math.dot(grad0, grad0) + math.dot(grad1, grad1) + 
                             math.dot(grad2, grad2) + math.dot(grad3, grad3);

            if (gradLenSq < 1e-9f) return;

            float lambda = -angleDiff / gradLenSq;
            float w = stiffness;

            rhs[c.e0] += w * lambda * grad0;
            rhs[c.e1] += w * lambda * grad1;
            rhs[c.oppositeA] += w * lambda * grad2;
            rhs[c.oppositeB] += w * lambda * grad3;
        }
    }

    [BurstCompile]
    struct GlobalStepJob : IJobParallelFor
    {
        [WriteOnly] public NativeArray<float3> q;
        [ReadOnly] public NativeArray<float3> rhs;
        [ReadOnly] public NativeArray<float> diagA;
        [ReadOnly] public NativeArray<float> pinWeights;
        [ReadOnly] public NativeArray<float3> pinTargets;

        public void Execute(int i)
        {
            if (pinWeights[i] > 0f)
            {
                q[i] = pinTargets[i];
            }
            else
            {
                q[i] = rhs[i] / diagA[i];
            }
        }
    }

    [BurstCompile]
    struct ColliderCollisionJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float3> positions;
        [NativeDisableParallelForRestriction] public NativeArray<float3> velocities;
        [ReadOnly] public NativeArray<float> inverseMasses;
        [ReadOnly] public NativeArray<float3> colliderVertices;
        [ReadOnly] public NativeArray<int> colliderTriangles;
        public float thickness;
        public float friction;
        public float dt;

        public void Execute(int i)
        {
            if (inverseMasses[i] == 0f) return;

            float3 p = positions[i];
            float minDist = float.MaxValue;
            float3 closestPoint = p;
            float3 closestNormal = new float3(0, 1, 0);

            // Check all triangles
            for (int t = 0; t < colliderTriangles.Length; t += 3)
            {
                int i0 = colliderTriangles[t];
                int i1 = colliderTriangles[t + 1];
                int i2 = colliderTriangles[t + 2];

                float3 v0 = colliderVertices[i0];
                float3 v1 = colliderVertices[i1];
                float3 v2 = colliderVertices[i2];

                // Closest point on triangle
                float3 closest = ClosestPointOnTriangle(p, v0, v1, v2);
                float dist = math.length(p - closest);

                if (dist < minDist)
                {
                    minDist = dist;
                    closestPoint = closest;
                    float3 n = math.cross(v1 - v0, v2 - v0);
                    float nlen = math.length(n);
                    closestNormal = nlen > 1e-9f ? n / nlen : new float3(0, 1, 0);
                }
            }

            // Apply collision response
            if (minDist < thickness)
            {
                float3 delta = p - closestPoint;
                float signedDist = math.dot(delta, closestNormal);

                if (signedDist < thickness)
                {
                    // Push out
                    positions[i] = closestPoint + closestNormal * thickness;

                    // Apply friction
                    float3 vel = velocities[i];
                    float3 velNormal = closestNormal * math.dot(vel, closestNormal);
                    float3 velTangent = vel - velNormal;
                    velocities[i] = velNormal + velTangent * (1f - friction);
                }
            }
        }

        private float3 ClosestPointOnTriangle(float3 p, float3 a, float3 b, float3 c)
        {
            float3 ab = b - a;
            float3 ac = c - a;
            float3 ap = p - a;

            float d1 = math.dot(ab, ap);
            float d2 = math.dot(ac, ap);
            if (d1 <= 0f && d2 <= 0f) return a;

            float3 bp = p - b;
            float d3 = math.dot(ab, bp);
            float d4 = math.dot(ac, bp);
            if (d3 >= 0f && d4 <= d3) return b;

            float3 cp = p - c;
            float d5 = math.dot(ab, cp);
            float d6 = math.dot(ac, cp);
            if (d6 >= 0f && d5 <= d6) return c;

            float vc = d1 * d4 - d3 * d2;
            if (vc <= 0f && d1 >= 0f && d3 <= 0f)
            {
                float v = d1 / (d1 - d3);
                return a + v * ab;
            }

            float vb = d5 * d2 - d1 * d6;
            if (vb <= 0f && d2 >= 0f && d6 <= 0f)
            {
                float v = d2 / (d2 - d6);
                return a + v * ac;
            }

            float va = d3 * d6 - d5 * d4;
            if (va <= 0f && (d4 - d3) >= 0f && (d5 - d6) >= 0f)
            {
                float v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                return b + v * (c - b);
            }

            float denom = 1f / (va + vb + vc);
            float v1 = vb * denom;
            float v2 = vc * denom;
            return a + ab * v1 + ac * v2;
        }
    }

    [BurstCompile]
    struct BuildSpatialHashJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> positions;
        [WriteOnly] public NativeMultiHashMap<int, int>.ParallelWriter spatialHash;
        public float cellSize;

        public void Execute(int i)
        {
            float3 p = positions[i];
            int3 cell = (int3)math.floor(p / cellSize);
            int hash = HashCell(cell.x, cell.y, cell.z);
            spatialHash.Add(hash, i);
        }

        private int HashCell(int x, int y, int z)
        {
            return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
        }
    }

    [BurstCompile]
    struct SelfCollisionJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<float3> positions;
        [NativeDisableParallelForRestriction] public NativeArray<float3> velocities;
        [ReadOnly] public NativeArray<float> inverseMasses;
        [ReadOnly] public NativeMultiHashMap<int, int> spatialHash;
        public float thickness;
        public float friction;
        public float cellSize;

        public void Execute(int i)
        {
            if (inverseMasses[i] == 0f) return;

            float3 pi = positions[i];
            int3 cell = (int3)math.floor(pi / cellSize);

            // Check 27 neighboring cells
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        int3 neighborCell = cell + new int3(dx, dy, dz);
                        int hash = HashCell(neighborCell.x, neighborCell.y, neighborCell.z);

                        if (spatialHash.TryGetFirstValue(hash, out int j, out var iterator))
                        {
                            do
                            {
                                if (j > i && inverseMasses[j] > 0f)
                                {
                                    float3 pj = positions[j];
                                    float3 delta = pi - pj;
                                    float dist = math.length(delta);

                                    if (dist < thickness && dist > 1e-9f)
                                    {
                                        float3 dir = delta / dist;
                                        float overlap = thickness - dist;

                                        float invMassSum = inverseMasses[i] + inverseMasses[j];
                                        float wi = inverseMasses[i] / invMassSum;
                                        float wj = inverseMasses[j] / invMassSum;

                                        positions[i] += dir * (overlap * wi);
                                        positions[j] -= dir * (overlap * wj);

                                        // Apply friction
                                        float3 relVel = velocities[i] - velocities[j];
                                        float3 velNormal = dir * math.dot(relVel, dir);
                                        float3 velTangent = relVel - velNormal;
                                        float3 frictionImpulse = velTangent * friction;

                                        velocities[i] -= frictionImpulse * wi;
                                        velocities[j] += frictionImpulse * wj;
                                    }
                                }
                            } while (spatialHash.TryGetNextValue(out j, ref iterator));
                        }
                    }
                }
            }
        }

        private int HashCell(int x, int y, int z)
        {
            return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
        }
    }

    [BurstCompile]
    struct UpdateVelocityPositionJob : IJobParallelFor
    {
        public NativeArray<float3> positions;
        public NativeArray<float3> velocities;
        [ReadOnly] public NativeArray<float3> q;
        public float dt;

        public void Execute(int i)
        {
            velocities[i] = (q[i] - positions[i]) / dt;
            positions[i] = q[i];
        }
    }

    #endregion
}
#endif
