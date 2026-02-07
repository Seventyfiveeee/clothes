#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using Unity.Jobs;
using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ClothBaking
{
    // ============================================================================
    // DATA STRUCTURES
    // ============================================================================

    [Serializable]
    public struct DistanceConstraint
    {
        public int particleA;
        public int particleB;
        public float restLength;
        public float weight;
    }

    [Serializable]
    public struct BendingConstraint
    {
        public int p0, p1, p2, p3; // edge (p0,p1), opposite vertices p2, p3
        public float Q00, Q01, Q02, Q03;
        public float Q11, Q12, Q13;
        public float Q22, Q23;
        public float Q33;
        public float weight;
    }

    [Serializable]
    public struct ParticleConstraintRef
    {
        public int constraintIndex;
        public int localIndex; // 0 or 1 (which end of the constraint)
    }

    [Serializable]
    public struct ParticleBendRef
    {
        public int bendIndex;
        public int localIndex; // 0,1,2,3
    }

    [Serializable]
    public struct SphereColliderData
    {
        public float3 center;
        public float radius;
    }

    [Serializable]
    public struct CapsuleColliderData
    {
        public float3 point0;
        public float3 point1;
        public float radius;
    }

    // ============================================================================
    // BURST JOBS
    // ============================================================================

    [BurstCompile]
    public struct MomentumStepJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<float3> velocities;
        [ReadOnly] public NativeArray<float> invMasses;
        public float dt;
        public float3 gravity;
        [WriteOnly] public NativeArray<float3> momentum;

        public void Execute(int i)
        {
            if (invMasses[i] == 0f)
            {
                momentum[i] = positions[i];
            }
            else
            {
                float3 acc = gravity;
                momentum[i] = positions[i] + velocities[i] * dt + acc * (dt * dt);
            }
        }
    }

    [BurstCompile]
    public struct CopyFloat3Job : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> source;
        [WriteOnly] public NativeArray<float3> dest;

        public void Execute(int i)
        {
            dest[i] = source[i];
        }
    }

    [BurstCompile]
    public struct ZeroFloat3Job : IJobParallelFor
    {
        [WriteOnly] public NativeArray<float3> array;

        public void Execute(int i)
        {
            array[i] = float3.zero;
        }
    }

    [BurstCompile]
    public struct ZeroFloatJob : IJobParallelFor
    {
        [WriteOnly] public NativeArray<float> array;

        public void Execute(int i)
        {
            array[i] = 0f;
        }
    }

    [BurstCompile]
    public struct LocalStepDistanceJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<float> invMasses;
        [ReadOnly] public NativeArray<DistanceConstraint> constraints;
        [WriteOnly] public NativeArray<float3> projections;

        public void Execute(int i)
        {
            var c = constraints[i];
            float3 pA = positions[c.particleA];
            float3 pB = positions[c.particleB];
            float3 diff = pB - pA;
            float dist = math.length(diff);

            if (dist < 1e-7f)
            {
                projections[i * 2] = pA;
                projections[i * 2 + 1] = pB;
                return;
            }

            float3 dir = diff / dist;
            float wA = invMasses[c.particleA];
            float wB = invMasses[c.particleB];
            float wSum = wA + wB;

            if (wSum < 1e-9f)
            {
                projections[i * 2] = pA;
                projections[i * 2 + 1] = pB;
                return;
            }

            float lambda = (dist - c.restLength) / wSum;
            projections[i * 2] = pA + dir * (lambda * wA);
            projections[i * 2 + 1] = pB - dir * (lambda * wB);
        }
    }

    [BurstCompile]
    public struct GlobalStepGatherAndSolveJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> momentum;
        [ReadOnly] public NativeArray<float> invMasses;
        [ReadOnly] public NativeArray<float> masses;
        public float dt;

        // Distance constraints
        [ReadOnly] public NativeArray<float3> distanceProjections;
        [ReadOnly] public NativeArray<DistanceConstraint> distanceConstraints;
        [ReadOnly] public NativeArray<int> distanceOffsets;
        [ReadOnly] public NativeArray<ParticleConstraintRef> distanceRefs;

        // Bending constraints (Jacobi splitting)
        [ReadOnly] public NativeArray<BendingConstraint> bendConstraints;
        [ReadOnly] public NativeArray<int> bendOffsets;
        [ReadOnly] public NativeArray<ParticleBendRef> bendRefs;
        [ReadOnly] public NativeArray<float3> currentPositions;

        [NativeDisableParallelForRestriction]
        public NativeArray<float3> newPositions;

        public void Execute(int i)
        {
            if (invMasses[i] == 0f)
            {
                newPositions[i] = momentum[i];
                return;
            }

            float massWeight = masses[i] / (dt * dt);
            float3 numerator = massWeight * momentum[i];
            float denominator = massWeight;

            // Distance constraints
            int distStart = distanceOffsets[i];
            int distEnd = distanceOffsets[i + 1];
            for (int j = distStart; j < distEnd; j++)
            {
                var cref = distanceRefs[j];
                var constraint = distanceConstraints[cref.constraintIndex];
                float3 proj = distanceProjections[cref.constraintIndex * 2 + cref.localIndex];
                numerator += constraint.weight * proj;
                denominator += constraint.weight;
            }

            // Bending constraints (Jacobi splitting)
            int bendStart = bendOffsets[i];
            int bendEnd = bendOffsets[i + 1];
            for (int j = bendStart; j < bendEnd; j++)
            {
                var bref = bendRefs[j];
                var bend = bendConstraints[bref.bendIndex];
                int localIdx = bref.localIndex;

                float Qii = 0f;
                if (localIdx == 0) Qii = bend.Q00;
                else if (localIdx == 1) Qii = bend.Q11;
                else if (localIdx == 2) Qii = bend.Q22;
                else if (localIdx == 3) Qii = bend.Q33;

                // Off-diagonal contribution to RHS
                float3 offDiag = float3.zero;
                int[] indices = { bend.p0, bend.p1, bend.p2, bend.p3 };
                for (int k = 0; k < 4; k++)
                {
                    if (k == localIdx) continue;
                    float Qik = GetQElement(bend, localIdx, k);
                    offDiag += Qik * currentPositions[indices[k]];
                }

                numerator -= bend.weight * offDiag;
                denominator += bend.weight * Qii;
            }

            if (denominator < 1e-9f)
            {
                newPositions[i] = momentum[i];
            }
            else
            {
                newPositions[i] = numerator / denominator;
            }
        }

        private float GetQElement(BendingConstraint bend, int i, int j)
        {
            if (i > j)
            {
                int tmp = i;
                i = j;
                j = tmp;
            }

            if (i == 0)
            {
                if (j == 0) return bend.Q00;
                if (j == 1) return bend.Q01;
                if (j == 2) return bend.Q02;
                if (j == 3) return bend.Q03;
            }
            else if (i == 1)
            {
                if (j == 1) return bend.Q11;
                if (j == 2) return bend.Q12;
                if (j == 3) return bend.Q13;
            }
            else if (i == 2)
            {
                if (j == 2) return bend.Q22;
                if (j == 3) return bend.Q23;
            }
            else if (i == 3)
            {
                return bend.Q33;
            }
            return 0f;
        }
    }

    [BurstCompile]
    public struct SelfCollisionJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<float> invMasses;
        public float thickness;
        public float hashCellSize;
        [ReadOnly] public NativeParallelMultiHashMap<int3, int> spatialHash;
        [WriteOnly] public NativeArray<float3> corrections;

        public void Execute(int i)
        {
            if (invMasses[i] == 0f)
            {
                corrections[i] = float3.zero;
                return;
            }

            float3 pos = positions[i];
            int3 cellCoord = (int3)math.floor(pos / hashCellSize);
            float3 totalCorrection = float3.zero;
            int collisionCount = 0;

            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        int3 neighborCell = cellCoord + new int3(dx, dy, dz);
                        if (spatialHash.TryGetFirstValue(neighborCell, out int j, out var it))
                        {
                            do
                            {
                                if (j == i) continue;
                                float3 diff = pos - positions[j];
                                float dist = math.length(diff);
                                if (dist < thickness && dist > 1e-7f)
                                {
                                    float penetration = thickness - dist;
                                    float3 dir = diff / dist;
                                    float wA = invMasses[i];
                                    float wB = invMasses[j];
                                    float wSum = wA + wB;
                                    if (wSum > 1e-9f)
                                    {
                                        totalCorrection += dir * (penetration * wA / wSum);
                                        collisionCount++;
                                    }
                                }
                            } while (spatialHash.TryGetNextValue(out j, ref it));
                        }
                    }
                }
            }

            corrections[i] = totalCorrection;
        }
    }

    [BurstCompile]
    public struct ApplyCollisionCorrectionsJob : IJobParallelFor
    {
        public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<float3> corrections;

        public void Execute(int i)
        {
            positions[i] += corrections[i];
        }
    }

    [BurstCompile]
    public struct SphereCollisionJob : IJobParallelFor
    {
        public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<float> invMasses;
        [ReadOnly] public NativeArray<SphereColliderData> spheres;

        public void Execute(int i)
        {
            if (invMasses[i] == 0f) return;

            float3 pos = positions[i];
            for (int s = 0; s < spheres.Length; s++)
            {
                var sphere = spheres[s];
                float3 diff = pos - sphere.center;
                float dist = math.length(diff);
                if (dist < sphere.radius && dist > 1e-7f)
                {
                    float3 dir = diff / dist;
                    positions[i] = sphere.center + dir * sphere.radius;
                }
            }
        }
    }

    [BurstCompile]
    public struct CapsuleCollisionJob : IJobParallelFor
    {
        public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<float> invMasses;
        [ReadOnly] public NativeArray<CapsuleColliderData> capsules;

        public void Execute(int i)
        {
            if (invMasses[i] == 0f) return;

            float3 pos = positions[i];
            for (int c = 0; c < capsules.Length; c++)
            {
                var capsule = capsules[c];
                float3 axis = capsule.point1 - capsule.point0;
                float axisLen = math.length(axis);
                if (axisLen < 1e-7f) continue;

                float3 axisDir = axis / axisLen;
                float3 diff = pos - capsule.point0;
                float t = math.clamp(math.dot(diff, axisDir), 0f, axisLen);
                float3 closest = capsule.point0 + axisDir * t;
                float3 toPos = pos - closest;
                float dist = math.length(toPos);

                if (dist < capsule.radius && dist > 1e-7f)
                {
                    float3 dir = toPos / dist;
                    positions[i] = closest + dir * capsule.radius;
                }
            }
        }
    }

    [BurstCompile]
    public struct VelocityUpdateJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> newPositions;
        [ReadOnly] public NativeArray<float3> oldPositions;
        public float damping;
        public float dt;
        public NativeArray<float3> velocities;

        public void Execute(int i)
        {
            velocities[i] = ((newPositions[i] - oldPositions[i]) / dt) * damping;
        }
    }

    // ============================================================================
    // CLOTH OFFLINE BAKER MONOBEHAVIOUR
    // ============================================================================

    [ExecuteInEditMode]
    public class ClothOfflineBaker : MonoBehaviour
    {
        [Header("Simulation Parameters")]
        public float timeStep = 0.008f;
        public int solverIterations = 50;
        public int totalFrames = 300;
        public float distanceWeight = 1000f;
        public float bendingStiffness = 0.1f;
        public float particleMassValue = 0.1f;
        public float damping = 0.995f;
        public Vector3 gravity = new Vector3(0, -9.81f, 0);

        [Header("Collision")]
        public float thickness = 0.005f;
        public bool enableSelfCollision = true;
        public float hashCellSize = 0.02f;
        public int collisionSubstepInterval = 5;

        [Header("Colliders")]
        public List<Transform> sphereColliders = new List<Transform>();
        public List<Transform> capsuleColliders = new List<Transform>();

        [Header("Pinning")]
        public List<int> pinnedParticleIndices = new List<int>();

        // Internal state
        private Mesh workingMesh;
        private int particleCount;
        private NativeArray<float3> positions;
        private NativeArray<float3> velocities;
        private NativeArray<float3> momentum;
        private NativeArray<float> masses;
        private NativeArray<float> invMasses;

        private NativeArray<DistanceConstraint> distanceConstraints;
        private NativeArray<BendingConstraint> bendingConstraints;
        private NativeArray<float3> distanceProjections;

        private NativeArray<int> distanceOffsets;
        private NativeArray<ParticleConstraintRef> distanceRefs;
        private NativeArray<int> bendOffsets;
        private NativeArray<ParticleBendRef> bendRefs;

        private NativeArray<float3> collisionCorrections;
        private NativeParallelMultiHashMap<int3, int> spatialHash;

        private NativeArray<SphereColliderData> sphereData;
        private NativeArray<CapsuleColliderData> capsuleData;

        private Vector3[] originalVertices;
        private int[] originalTriangles;
        private Vector3[] originalNormals;
        private Vector4[] originalTangents;

        private bool isInitialised = false;
        private int currentFrame = 0;

        // ========================================================================
        // INITIALISATION
        // ========================================================================

        [ContextMenu("Initialise")]
        public void Initialise()
        {
            Cleanup();

            MeshFilter mf = GetComponent<MeshFilter>();
            if (mf == null || mf.sharedMesh == null)
            {
                Debug.LogError("No MeshFilter or mesh found!");
                return;
            }

            Mesh sourceMesh = mf.sharedMesh;
            workingMesh = Instantiate(sourceMesh);

            originalVertices = sourceMesh.vertices;
            originalTriangles = sourceMesh.triangles;
            originalNormals = sourceMesh.normals;
            originalTangents = sourceMesh.tangents;

            particleCount = originalVertices.Length;

            // Transform to world space
            Vector3[] worldVerts = new Vector3[particleCount];
            for (int i = 0; i < particleCount; i++)
            {
                worldVerts[i] = transform.TransformPoint(originalVertices[i]);
            }

            // Allocate arrays
            positions = new NativeArray<float3>(particleCount, Allocator.Persistent);
            velocities = new NativeArray<float3>(particleCount, Allocator.Persistent);
            momentum = new NativeArray<float3>(particleCount, Allocator.Persistent);
            masses = new NativeArray<float>(particleCount, Allocator.Persistent);
            invMasses = new NativeArray<float>(particleCount, Allocator.Persistent);

            for (int i = 0; i < particleCount; i++)
            {
                positions[i] = worldVerts[i];
                velocities[i] = float3.zero;
                masses[i] = particleMassValue;
                invMasses[i] = pinnedParticleIndices.Contains(i) ? 0f : 1f / particleMassValue;
            }

            // Build distance constraints from edges
            BuildDistanceConstraints(originalTriangles);

            // Build bending constraints
            BuildBendingConstraints(originalTriangles, worldVerts);

            // Build CSR structures
            BuildCSR();

            // Allocate collision arrays
            collisionCorrections = new NativeArray<float3>(particleCount, Allocator.Persistent);
            spatialHash = new NativeParallelMultiHashMap<int3, int>(particleCount * 27, Allocator.Persistent);

            // Allocate collider arrays
            sphereData = new NativeArray<SphereColliderData>(sphereColliders.Count, Allocator.Persistent);
            capsuleData = new NativeArray<CapsuleColliderData>(capsuleColliders.Count, Allocator.Persistent);

            isInitialised = true;
            currentFrame = 0;

            Debug.Log($"Initialised cloth simulation: {particleCount} particles, " +
                     $"{distanceConstraints.Length} distance constraints, " +
                     $"{bendingConstraints.Length} bending constraints");
        }

        private void BuildDistanceConstraints(int[] triangles)
        {
            var edgeSet = new HashSet<(int, int)>();
            var edgeList = new List<DistanceConstraint>();

            for (int i = 0; i < triangles.Length; i += 3)
            {
                int a = triangles[i];
                int b = triangles[i + 1];
                int c = triangles[i + 2];

                AddEdge(edgeSet, edgeList, a, b);
                AddEdge(edgeSet, edgeList, b, c);
                AddEdge(edgeSet, edgeList, c, a);
            }

            distanceConstraints = new NativeArray<DistanceConstraint>(edgeList.Count, Allocator.Persistent);
            for (int i = 0; i < edgeList.Count; i++)
            {
                distanceConstraints[i] = edgeList[i];
            }

            distanceProjections = new NativeArray<float3>(edgeList.Count * 2, Allocator.Persistent);
        }

        private void AddEdge(HashSet<(int, int)> edgeSet, List<DistanceConstraint> edgeList, int a, int b)
        {
            if (a > b)
            {
                int tmp = a;
                a = b;
                b = tmp;
            }

            if (!edgeSet.Contains((a, b)))
            {
                edgeSet.Add((a, b));
                float3 pA = positions[a];
                float3 pB = positions[b];
                float restLen = math.length(pB - pA);
                edgeList.Add(new DistanceConstraint
                {
                    particleA = a,
                    particleB = b,
                    restLength = restLen,
                    weight = distanceWeight
                });
            }
        }

        private void BuildBendingConstraints(int[] triangles, Vector3[] worldVerts)
        {
            // Build edge-to-triangle adjacency
            var edgeToTris = new Dictionary<(int, int), List<int>>();

            for (int triIdx = 0; triIdx < triangles.Length / 3; triIdx++)
            {
                int i0 = triangles[triIdx * 3];
                int i1 = triangles[triIdx * 3 + 1];
                int i2 = triangles[triIdx * 3 + 2];

                AddTriToEdge(edgeToTris, i0, i1, triIdx);
                AddTriToEdge(edgeToTris, i1, i2, triIdx);
                AddTriToEdge(edgeToTris, i2, i0, triIdx);
            }

            // Find interior edges (shared by 2 triangles)
            var bendList = new List<BendingConstraint>();
            foreach (var kvp in edgeToTris)
            {
                if (kvp.Value.Count == 2)
                {
                    var edge = kvp.Key;
                    int p0 = edge.Item1;
                    int p1 = edge.Item2;
                    int tri0 = kvp.Value[0];
                    int tri1 = kvp.Value[1];

                    int p2 = GetOppositeVertex(triangles, tri0, p0, p1);
                    int p3 = GetOppositeVertex(triangles, tri1, p0, p1);

                    if (p2 == -1 || p3 == -1) continue;

                    var Q = ComputeIsometricBendingMatrix(worldVerts, p0, p1, p2, p3);
                    if (Q.HasValue)
                    {
                        bendList.Add(Q.Value);
                    }
                }
            }

            bendingConstraints = new NativeArray<BendingConstraint>(bendList.Count, Allocator.Persistent);
            for (int i = 0; i < bendList.Count; i++)
            {
                bendingConstraints[i] = bendList[i];
            }
        }

        private void AddTriToEdge(Dictionary<(int, int), List<int>> edgeToTris, int a, int b, int triIdx)
        {
            if (a > b)
            {
                int tmp = a;
                a = b;
                b = tmp;
            }

            if (!edgeToTris.ContainsKey((a, b)))
            {
                edgeToTris[(a, b)] = new List<int>();
            }
            edgeToTris[(a, b)].Add(triIdx);
        }

        private int GetOppositeVertex(int[] triangles, int triIdx, int e0, int e1)
        {
            int i0 = triangles[triIdx * 3];
            int i1 = triangles[triIdx * 3 + 1];
            int i2 = triangles[triIdx * 3 + 2];

            if (i0 != e0 && i0 != e1) return i0;
            if (i1 != e0 && i1 != e1) return i1;
            if (i2 != e0 && i2 != e1) return i2;
            return -1;
        }

        private BendingConstraint? ComputeIsometricBendingMatrix(Vector3[] verts, int p0, int p1, int p2, int p3)
        {
            float3 x0 = verts[p0];
            float3 x1 = verts[p1];
            float3 x2 = verts[p2];
            float3 x3 = verts[p3];

            float3 e = x1 - x0;
            float3 n1 = math.cross(e, x2 - x0);
            float3 n2 = math.cross(x3 - x0, e);

            float A1 = math.length(n1) * 0.5f;
            float A2 = math.length(n2) * 0.5f;

            if (A1 < 1e-9f || A2 < 1e-9f) return null;

            // Cotangent weights
            float cot01 = Cot(x2 - x0, x2 - x1);
            float cot02 = Cot(x3 - x0, x3 - x1);

            float A = A1 + A2;
            float scale = 3.0f / A;

            // K = [cot01 + cot02, -cot01 - cot02, -cot01, -cot02]
            float4 K = new float4(cot01 + cot02, -cot01 - cot02, -cot01, -cot02);

            // Q = scale * K * K^T
            float Q00 = scale * K.x * K.x;
            float Q01 = scale * K.x * K.y;
            float Q02 = scale * K.x * K.z;
            float Q03 = scale * K.x * K.w;
            float Q11 = scale * K.y * K.y;
            float Q12 = scale * K.y * K.z;
            float Q13 = scale * K.y * K.w;
            float Q22 = scale * K.z * K.z;
            float Q23 = scale * K.z * K.w;
            float Q33 = scale * K.w * K.w;

            return new BendingConstraint
            {
                p0 = p0,
                p1 = p1,
                p2 = p2,
                p3 = p3,
                Q00 = Q00,
                Q01 = Q01,
                Q02 = Q02,
                Q03 = Q03,
                Q11 = Q11,
                Q12 = Q12,
                Q13 = Q13,
                Q22 = Q22,
                Q23 = Q23,
                Q33 = Q33,
                weight = bendingStiffness
            };
        }

        private float Cot(float3 a, float3 b)
        {
            float cosTheta = math.dot(a, b);
            float sinTheta = math.length(math.cross(a, b));
            if (math.abs(sinTheta) < 1e-9f) return 0f;
            return cosTheta / sinTheta;
        }

        private void BuildCSR()
        {
            // Build distance constraint CSR
            var distPerParticle = new List<ParticleConstraintRef>[particleCount];
            for (int i = 0; i < particleCount; i++)
            {
                distPerParticle[i] = new List<ParticleConstraintRef>();
            }

            for (int i = 0; i < distanceConstraints.Length; i++)
            {
                var c = distanceConstraints[i];
                distPerParticle[c.particleA].Add(new ParticleConstraintRef { constraintIndex = i, localIndex = 0 });
                distPerParticle[c.particleB].Add(new ParticleConstraintRef { constraintIndex = i, localIndex = 1 });
            }

            int totalDistRefs = distPerParticle.Sum(list => list.Count);
            distanceOffsets = new NativeArray<int>(particleCount + 1, Allocator.Persistent);
            distanceRefs = new NativeArray<ParticleConstraintRef>(totalDistRefs, Allocator.Persistent);

            int offset = 0;
            for (int i = 0; i < particleCount; i++)
            {
                distanceOffsets[i] = offset;
                foreach (var cref in distPerParticle[i])
                {
                    distanceRefs[offset++] = cref;
                }
            }
            distanceOffsets[particleCount] = offset;

            // Build bending constraint CSR
            var bendPerParticle = new List<ParticleBendRef>[particleCount];
            for (int i = 0; i < particleCount; i++)
            {
                bendPerParticle[i] = new List<ParticleBendRef>();
            }

            for (int i = 0; i < bendingConstraints.Length; i++)
            {
                var b = bendingConstraints[i];
                bendPerParticle[b.p0].Add(new ParticleBendRef { bendIndex = i, localIndex = 0 });
                bendPerParticle[b.p1].Add(new ParticleBendRef { bendIndex = i, localIndex = 1 });
                bendPerParticle[b.p2].Add(new ParticleBendRef { bendIndex = i, localIndex = 2 });
                bendPerParticle[b.p3].Add(new ParticleBendRef { bendIndex = i, localIndex = 3 });
            }

            int totalBendRefs = bendPerParticle.Sum(list => list.Count);
            bendOffsets = new NativeArray<int>(particleCount + 1, Allocator.Persistent);
            bendRefs = new NativeArray<ParticleBendRef>(totalBendRefs, Allocator.Persistent);

            offset = 0;
            for (int i = 0; i < particleCount; i++)
            {
                bendOffsets[i] = offset;
                foreach (var bref in bendPerParticle[i])
                {
                    bendRefs[offset++] = bref;
                }
            }
            bendOffsets[particleCount] = offset;
        }

        // ========================================================================
        // SIMULATION STEP
        // ========================================================================

        [ContextMenu("Bake Single Frame")]
        public void BakeSingleFrame()
        {
            if (!isInitialised)
            {
                Debug.LogWarning("Not initialised. Call Initialise first.");
                return;
            }

            SimulateStep();
            WriteMesh();
            currentFrame++;

            Debug.Log($"Baked frame {currentFrame}");
        }

        [ContextMenu("Bake All Frames")]
        public void BakeAllFrames()
        {
            if (!isInitialised)
            {
                Debug.LogWarning("Not initialised. Call Initialise first.");
                return;
            }

            for (int i = 0; i < totalFrames; i++)
            {
                SimulateStep();
                WriteMesh();
                currentFrame++;

                if (i % 10 == 0)
                {
                    Debug.Log($"Baked frame {currentFrame}/{totalFrames}");
                }
            }

            Debug.Log($"Baking complete: {totalFrames} frames");
        }

        private void SimulateStep()
        {
            // 1. Momentum step
            var momentumJob = new MomentumStepJob
            {
                positions = positions,
                velocities = velocities,
                invMasses = invMasses,
                dt = timeStep,
                gravity = gravity,
                momentum = momentum
            };
            momentumJob.Schedule(particleCount, 64).Complete();

            // Save old positions for velocity update
            NativeArray<float3> oldPositions = new NativeArray<float3>(particleCount, Allocator.TempJob);
            var copyJob = new CopyFloat3Job { source = positions, dest = oldPositions };
            copyJob.Schedule(particleCount, 64).Complete();

            // 2. Local-global iterations with interleaved collision
            for (int iter = 0; iter < solverIterations; iter++)
            {
                // Local step: distance constraints
                var localJob = new LocalStepDistanceJob
                {
                    positions = positions,
                    invMasses = invMasses,
                    constraints = distanceConstraints,
                    projections = distanceProjections
                };
                localJob.Schedule(distanceConstraints.Length, 64).Complete();

                // Global step: gather and solve
                var globalJob = new GlobalStepGatherAndSolveJob
                {
                    momentum = momentum,
                    invMasses = invMasses,
                    masses = masses,
                    dt = timeStep,
                    distanceProjections = distanceProjections,
                    distanceConstraints = distanceConstraints,
                    distanceOffsets = distanceOffsets,
                    distanceRefs = distanceRefs,
                    bendConstraints = bendingConstraints,
                    bendOffsets = bendOffsets,
                    bendRefs = bendRefs,
                    currentPositions = positions,
                    newPositions = positions
                };
                globalJob.Schedule(particleCount, 64).Complete();

                // Interleaved collision
                if (enableSelfCollision && (iter + 1) % collisionSubstepInterval == 0)
                {
                    ResolveSelfCollision();
                }

                ResolveSphereCollisions();
                ResolveCapsuleCollisions();
            }

            // Final collision pass
            if (enableSelfCollision)
            {
                ResolveSelfCollision();
            }
            ResolveSphereCollisions();
            ResolveCapsuleCollisions();

            // 3. Velocity update
            var velocityJob = new VelocityUpdateJob
            {
                newPositions = positions,
                oldPositions = oldPositions,
                damping = damping,
                dt = timeStep,
                velocities = velocities
            };
            velocityJob.Schedule(particleCount, 64).Complete();

            oldPositions.Dispose();
        }

        private void ResolveSelfCollision()
        {
            // Build spatial hash
            spatialHash.Clear();
            for (int i = 0; i < particleCount; i++)
            {
                int3 cellCoord = (int3)math.floor(positions[i] / hashCellSize);
                spatialHash.Add(cellCoord, i);
            }

            // Compute corrections
            var collisionJob = new SelfCollisionJob
            {
                positions = positions,
                invMasses = invMasses,
                thickness = thickness,
                hashCellSize = hashCellSize,
                spatialHash = spatialHash,
                corrections = collisionCorrections
            };
            collisionJob.Schedule(particleCount, 64).Complete();

            // Apply corrections
            var applyJob = new ApplyCollisionCorrectionsJob
            {
                positions = positions,
                corrections = collisionCorrections
            };
            applyJob.Schedule(particleCount, 64).Complete();
        }

        private void ResolveSphereCollisions()
        {
            if (sphereColliders.Count == 0) return;

            // Update sphere data
            int validCount = 0;
            for (int i = 0; i < sphereColliders.Count; i++)
            {
                var t = sphereColliders[i];
                if (t == null) continue;

                SphereCollider sc = t.GetComponent<SphereCollider>();
                if (sc != null)
                {
                    float3 center = t.TransformPoint(sc.center);
                    float radius = sc.radius * math.cmax(t.lossyScale);
                    sphereData[validCount++] = new SphereColliderData { center = center, radius = radius };
                }
                else
                {
                    // Fallback: use transform position with default radius
                    float3 center = t.position;
                    float radius = 0.1f * math.cmax(t.lossyScale);
                    sphereData[validCount++] = new SphereColliderData { center = center, radius = radius };
                }
            }

            if (validCount == 0) return;

            var sphereJob = new SphereCollisionJob
            {
                positions = positions,
                invMasses = invMasses,
                spheres = sphereData.GetSubArray(0, validCount)
            };
            sphereJob.Schedule(particleCount, 64).Complete();
        }

        private void ResolveCapsuleCollisions()
        {
            if (capsuleColliders.Count == 0) return;

            // Update capsule data
            int validCount = 0;
            for (int i = 0; i < capsuleColliders.Count; i++)
            {
                var t = capsuleColliders[i];
                if (t == null) continue;

                CapsuleCollider cc = t.GetComponent<CapsuleCollider>();
                if (cc != null)
                {
                    float3 center = t.TransformPoint(cc.center);
                    float height = cc.height * math.cmax(t.lossyScale);
                    float radius = cc.radius * math.cmax(t.lossyScale);

                    // Compute capsule axis
                    float3 axis = float3.zero;
                    if (cc.direction == 0) axis = new float3(1, 0, 0);
                    else if (cc.direction == 1) axis = new float3(0, 1, 0);
                    else axis = new float3(0, 0, 1);

                    axis = math.rotate(t.rotation, axis);
                    float halfHeight = math.max(0, (height - 2 * radius) * 0.5f);

                    float3 p0 = center - axis * halfHeight;
                    float3 p1 = center + axis * halfHeight;

                    capsuleData[validCount++] = new CapsuleColliderData
                    {
                        point0 = p0,
                        point1 = p1,
                        radius = radius
                    };
                }
            }

            if (validCount == 0) return;

            var capsuleJob = new CapsuleCollisionJob
            {
                positions = positions,
                invMasses = invMasses,
                capsules = capsuleData.GetSubArray(0, validCount)
            };
            capsuleJob.Schedule(particleCount, 64).Complete();
        }

        private void WriteMesh()
        {
            Vector3[] newVertices = new Vector3[particleCount];
            for (int i = 0; i < particleCount; i++)
            {
                float3 worldPos = positions[i];
                newVertices[i] = transform.InverseTransformPoint(worldPos);
            }

            workingMesh.vertices = newVertices;
            workingMesh.RecalculateNormals();
            workingMesh.RecalculateBounds();
            workingMesh.RecalculateTangents();

            MeshFilter mf = GetComponent<MeshFilter>();
            if (mf != null)
            {
                mf.mesh = workingMesh;
            }
        }

        // ========================================================================
        // UTILITY METHODS
        // ========================================================================

        [ContextMenu("Reset Simulation")]
        public void ResetSimulation()
        {
            if (!isInitialised)
            {
                Debug.LogWarning("Not initialised.");
                return;
            }

            for (int i = 0; i < particleCount; i++)
            {
                positions[i] = transform.TransformPoint(originalVertices[i]);
                velocities[i] = float3.zero;
            }

            currentFrame = 0;
            WriteMesh();

            Debug.Log("Simulation reset");
        }

        [ContextMenu("Save Baked Mesh")]
        public void SaveBakedMesh()
        {
            if (workingMesh == null)
            {
                Debug.LogWarning("No mesh to save.");
                return;
            }

            string path = EditorUtility.SaveFilePanelInProject(
                "Save Baked Mesh",
                "BakedClothMesh",
                "asset",
                "Save the baked cloth mesh as an asset");

            if (string.IsNullOrEmpty(path)) return;

            Mesh savedMesh = Instantiate(workingMesh);
            AssetDatabase.CreateAsset(savedMesh, path);
            AssetDatabase.SaveAssets();

            Debug.Log($"Saved mesh to {path}");
        }

        public void Cleanup()
        {
            if (positions.IsCreated) positions.Dispose();
            if (velocities.IsCreated) velocities.Dispose();
            if (momentum.IsCreated) momentum.Dispose();
            if (masses.IsCreated) masses.Dispose();
            if (invMasses.IsCreated) invMasses.Dispose();
            if (distanceConstraints.IsCreated) distanceConstraints.Dispose();
            if (bendingConstraints.IsCreated) bendingConstraints.Dispose();
            if (distanceProjections.IsCreated) distanceProjections.Dispose();
            if (distanceOffsets.IsCreated) distanceOffsets.Dispose();
            if (distanceRefs.IsCreated) distanceRefs.Dispose();
            if (bendOffsets.IsCreated) bendOffsets.Dispose();
            if (bendRefs.IsCreated) bendRefs.Dispose();
            if (collisionCorrections.IsCreated) collisionCorrections.Dispose();
            if (spatialHash.IsCreated) spatialHash.Dispose();
            if (sphereData.IsCreated) sphereData.Dispose();
            if (capsuleData.IsCreated) capsuleData.Dispose();

            isInitialised = false;
        }

        private void OnDisable()
        {
            Cleanup();
        }

        private void OnDestroy()
        {
            Cleanup();
        }
    }

    // ============================================================================
    // CUSTOM EDITOR
    // ============================================================================

    [CustomEditor(typeof(ClothOfflineBaker))]
    public class ClothOfflineBakerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            EditorGUILayout.Space();
            EditorGUILayout.HelpBox(
                "Cloth Offline Baker using Projective Dynamics with isometric bending.\n\n" +
                "1. Initialise - Setup simulation from mesh\n" +
                "2. Bake Single Frame - Simulate one frame\n" +
                "3. Bake All Frames - Simulate all frames\n" +
                "4. Reset - Return to initial state\n" +
                "5. Save Mesh Asset - Save current mesh as asset",
                MessageType.Info);

            EditorGUILayout.Space();

            ClothOfflineBaker baker = (ClothOfflineBaker)target;

            GUI.backgroundColor = Color.green;
            if (GUILayout.Button("Initialise", GUILayout.Height(40)))
            {
                baker.Initialise();
                SceneView.RepaintAll();
            }

            GUI.backgroundColor = Color.cyan;
            if (GUILayout.Button("Bake Single Frame", GUILayout.Height(40)))
            {
                baker.BakeSingleFrame();
                SceneView.RepaintAll();
            }

            GUI.backgroundColor = Color.blue;
            if (GUILayout.Button("Bake All Frames", GUILayout.Height(40)))
            {
                baker.BakeAllFrames();
                SceneView.RepaintAll();
            }

            GUI.backgroundColor = Color.yellow;
            if (GUILayout.Button("Reset Simulation", GUILayout.Height(40)))
            {
                baker.ResetSimulation();
                SceneView.RepaintAll();
            }

            GUI.backgroundColor = Color.magenta;
            if (GUILayout.Button("Save Mesh Asset", GUILayout.Height(40)))
            {
                baker.SaveBakedMesh();
            }

            GUI.backgroundColor = Color.white;
        }
    }
}

#endif
