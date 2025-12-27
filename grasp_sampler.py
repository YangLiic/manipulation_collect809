"""
Antipodal Grasp Sampler for Isaac Sim 4.5
Implements mesh-based grasp pose generation using trimesh library
"""

import numpy as np
import trimesh
from typing import List, Tuple, Optional
import omni.usd
from pxr import UsdGeom, Gf


class GraspPose:
    """Represents a 6-DOF grasp pose with quality score"""
    
    def __init__(self, position: np.ndarray, orientation: np.ndarray, quality: float = 0.0):
        """
        Args:
            position: 3D position (x, y, z)
            orientation: Quaternion (w, x, y, z)
            quality: Grasp quality score [0, 1]
        """
        self.position = position
        self.orientation = orientation  # quaternion [w, x, y, z]
        self.quality = quality
    
    def __repr__(self):
        return f"GraspPose(pos={self.position}, quat={self.orientation}, quality={self.quality:.3f})"


class AntipodalGraspSampler:
    """
    Samples antipodal grasp poses from object mesh
    Based on the approach used in Isaac Sim 5.x isaacsim.replicator.grasping
    """
    
    def __init__(
        self,
        gripper_width: float = 0.08,  # Franka gripper max width ~8cm
        standoff_distance: float = 0.02,  # Distance from TCP to object surface
        min_grasp_width: float = 0.01,  # Minimum object width to grasp
        approach_axis: str = 'z',  # Gripper approach direction in local frame
    ):
        """
        Initialize the antipodal grasp sampler
        
        Args:
            gripper_width: Maximum gripper aperture (meters)
            standoff_distance: Distance from gripper TCP to object surface
            min_grasp_width: Minimum width between antipodal points
            approach_axis: Gripper approach direction ('x', 'y', or 'z')
        """
        self.gripper_width = gripper_width
        self.standoff_distance = standoff_distance
        self.min_grasp_width = min_grasp_width
        self.approach_axis = approach_axis
        
        self.mesh: Optional[trimesh.Trimesh] = None
        self.candidate_grasps: List[GraspPose] = []
    
    def load_mesh_from_usd(self, prim_path: str) -> bool:
        """
        Extract mesh from USD stage and convert to trimesh
        
        Args:
            prim_path: USD path to the object prim (can be Xform or Mesh)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            
            if not prim or not prim.IsValid():
                print(f"❌ Invalid prim path: {prim_path}")
                return False
            
            # Try to get mesh from prim directly
            mesh_prim = UsdGeom.Mesh(prim)
            
            # If not a mesh, search for mesh children
            if not mesh_prim or not mesh_prim.GetPointsAttr():
                print(f"🔍 Prim is not a mesh, searching for mesh children...")
                mesh_prim = None
                
                # Recursively search for mesh prims
                def find_mesh(p):
                    if UsdGeom.Mesh(p) and UsdGeom.Mesh(p).GetPointsAttr():
                        return UsdGeom.Mesh(p)
                    for child in p.GetChildren():
                        result = find_mesh(child)
                        if result:
                            return result
                    return None
                
                mesh_prim = find_mesh(prim)
                
                if not mesh_prim:
                    print(f"❌ No mesh found under: {prim_path}")
                    return False
                else:
                    print(f"✅ Found mesh: {mesh_prim.GetPath()}")
            
            # Get vertices and faces
            points_attr = mesh_prim.GetPointsAttr()
            faces_attr = mesh_prim.GetFaceVertexIndicesAttr()
            face_counts_attr = mesh_prim.GetFaceVertexCountsAttr()
            
            if not points_attr or not faces_attr:
                print(f"❌ Mesh has no geometry data")
                return False
            
            # Read geometry data
            vertices = np.array(points_attr.Get())
            face_indices = np.array(faces_attr.Get())
            face_counts = np.array(face_counts_attr.Get()) if face_counts_attr else None
            
            # Convert to triangles (assuming triangulated mesh)
            if face_counts is not None and not np.all(face_counts == 3):
                print(f"⚠️ Mesh is not triangulated, attempting conversion...")
                # Simple triangulation for quads (not robust for general polygons)
                faces = []
                idx = 0
                for count in face_counts:
                    if count == 3:
                        faces.append(face_indices[idx:idx+3])
                    elif count == 4:
                        # Split quad into two triangles
                        quad = face_indices[idx:idx+4]
                        faces.append([quad[0], quad[1], quad[2]])
                        faces.append([quad[0], quad[2], quad[3]])
                    idx += count
                faces = np.array(faces)
            else:
                faces = face_indices.reshape(-1, 3)
            
            # Create trimesh object
            self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # 🔑 Apply prim's scale transform to get actual world size
            # Mesh vertices are in local coordinates, need to apply scale
            try:
                from pxr import Gf
                # Try to get scale from parent Xform
                parent_prim = prim.GetParent()
                scale_array = np.array([1.0, 1.0, 1.0])  # Default no scale
                
                # Check parent for scale (common pattern in USD)
                if parent_prim:
                    xformable = UsdGeom.Xformable(parent_prim)
                    if xformable:
                        # Get transform ops
                        xform_ops = xformable.GetOrderedXformOps()
                        for op in xform_ops:
                            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                                scale_val = op.Get()
                                if scale_val:
                                    scale_array = np.array([scale_val[0], scale_val[1], scale_val[2]])
                                    print(f"📏 Found USD scale from parent: {scale_array}")
                                    break
                
                # If no scale found, try to infer from bounding box vs world size
                # This is a fallback - we'll check if mesh is too large
                if np.allclose(scale_array, [1.0, 1.0, 1.0]):
                    print(f"⚠️ No explicit scale found, will check size heuristics")
                else:
                    self.mesh.vertices = self.mesh.vertices * scale_array
                    print(f"✅ Applied scale, new extents: {self.mesh.extents}")
                    
            except Exception as e:
                print(f"⚠️ Could not apply scale transform: {e}")
            
            # 🔑 Check if mesh is in millimeters (common USD issue)
            # If bounding box is huge (> 1 meter), likely in mm
            if np.max(self.mesh.extents) > 1.0:
                print(f"⚠️ Mesh appears to be in millimeters (extents: {self.mesh.extents}), converting to meters...")
                self.mesh.vertices = self.mesh.vertices / 1000.0  # Convert mm to m
                print(f"✅ Converted to meters (new extents: {self.mesh.extents})")
            
            # 🔑 Final sanity check: if still too large, apply additional scaling
            # Based on comment in working script: vegetable height ~ 0.05m (5cm)
            # If largest dimension > 0.5m, likely needs more scaling
            if np.max(self.mesh.extents) > 0.5:
                # Assume largest dimension should be ~0.05-0.10m
                current_max = np.max(self.mesh.extents)
                target_max = 0.08  # 8cm - reasonable for vegetable
                scale_factor = target_max / current_max
                print(f"⚠️ Mesh still too large (max: {current_max:.3f}m), applying heuristic scale: {scale_factor:.4f}")
                self.mesh.vertices = self.mesh.vertices * scale_factor
                print(f"✅ Final extents after heuristic scaling: {self.mesh.extents}")
            
            # Validate mesh
            if not self.mesh.is_watertight:
                print(f"⚠️ Mesh is not watertight, grasp quality may be affected")
            
            print(f"✅ Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
            print(f"📦 Final mesh extents: {self.mesh.extents} meters")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load mesh from USD: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def sample_antipodal_grasps(
        self,
        num_samples: int = 50,
        num_orientations: int = 4,
        random_seed: Optional[int] = None
    ) -> List[GraspPose]:
        """
        Generate antipodal grasp candidates
        
        Args:
            num_samples: Number of surface points to sample
            num_orientations: Rotational variations per grasp axis
            random_seed: Random seed for reproducibility
            
        Returns:
            List of candidate grasp poses
        """
        if self.mesh is None:
            print("❌ No mesh loaded. Call load_mesh_from_usd() first.")
            return []
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.candidate_grasps = []
        
        # Sample points on mesh surface
        points, face_indices = trimesh.sample.sample_surface(self.mesh, num_samples)
        
        # Get normals at sampled points
        normals = self.mesh.face_normals[face_indices]
        
        print(f"🔍 Sampled {len(points)} surface points")
        
        # Method 1: Ray casting for antipodal pairs
        antipodal_found = 0
        for i, (point1, normal1) in enumerate(zip(points, normals)):
            # Cast ray in opposite direction of normal to find antipodal point
            # Increase offset to avoid self-intersection
            ray_origin = point1 + normal1 * 0.01  # Larger offset
            ray_direction = -normal1
            
            # Ray-mesh intersection with longer ray
            locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                ray_origins=[ray_origin],
                ray_directions=[ray_direction]
            )
            
            if len(locations) == 0:
                continue
            
            # Use closest intersection as antipodal point
            point2 = locations[0]
            normal2 = self.mesh.face_normals[index_tri[0]]
            
            # Calculate grasp width (distance between antipodal points)
            grasp_width = np.linalg.norm(point2 - point1)
            
            # Filter by gripper constraints
            if grasp_width < self.min_grasp_width or grasp_width > self.gripper_width:
                continue
            
            antipodal_found += 1
            
            # Calculate grasp center and approach direction
            grasp_center = (point1 + point2) / 2.0
            approach_dir = normal1  # Approach from point1's normal direction
            
            # Generate multiple orientations around approach axis
            for rot_idx in range(num_orientations):
                angle = (2 * np.pi * rot_idx) / num_orientations
                
                # Create rotation matrix for grasp pose
                # Z-axis: approach direction
                # X-axis: perpendicular to approach (rotated)
                # Y-axis: cross product
                z_axis = approach_dir / np.linalg.norm(approach_dir)
                
                # Find perpendicular vector
                if abs(z_axis[2]) < 0.9:
                    x_temp = np.array([0, 0, 1])
                else:
                    x_temp = np.array([1, 0, 0])
                
                y_axis = np.cross(z_axis, x_temp)
                y_axis = y_axis / np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                
                # Rotate around z-axis
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                x_axis_rot = cos_a * x_axis + sin_a * y_axis
                y_axis_rot = -sin_a * x_axis + cos_a * y_axis
                
                # Build rotation matrix
                rotation_matrix = np.column_stack([x_axis_rot, y_axis_rot, z_axis])
                
                # Convert to quaternion [w, x, y, z]
                quat = self._rotation_matrix_to_quaternion(rotation_matrix)
                
                # Apply standoff distance
                grasp_position = grasp_center - z_axis * self.standoff_distance
                
                # Calculate grasp quality (simple metric based on alignment)
                quality = self._calculate_grasp_quality(normal1, normal2, grasp_width)
                
                # Create grasp pose
                grasp = GraspPose(
                    position=grasp_position,
                    orientation=quat,
                    quality=quality
                )
                
                self.candidate_grasps.append(grasp)
        
        print(f"✅ Found {antipodal_found} antipodal pairs via ray casting")
        
        # Method 2: Fallback - Bounding box based sampling if ray casting failed
        if len(self.candidate_grasps) == 0:
            print("⚠️ Ray casting failed, using bounding box sampling...")
            self._sample_from_bounding_box(num_orientations)
        
        print(f"✅ Generated {len(self.candidate_grasps)} candidate grasps")
        
        # Sort by quality (descending)
        self.candidate_grasps.sort(key=lambda g: g.quality, reverse=True)
        
        return self.candidate_grasps
    
    def _sample_from_bounding_box(self, num_orientations: int = 4):
        """
        Fallback method: Sample grasps based on object bounding box
        Generates grasps approaching from different directions
        """
        # Get bounding box
        bbox_center = self.mesh.centroid
        bbox_extents = self.mesh.extents  # [x, y, z] dimensions
        
        print(f"📦 Object center: {bbox_center}, extents: {bbox_extents}")
        
        # 🔑 Analyze object orientation in XY plane using PCA
        xy_vertices = self.mesh.vertices[:, :2]
        xy_centered = xy_vertices - np.mean(xy_vertices, axis=0)
        cov_matrix = np.cov(xy_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        principal_axis_idx = np.argmax(eigenvalues)
        principal_axis_xy = eigenvectors[:, principal_axis_idx]
        object_angle = np.arctan2(principal_axis_xy[1], principal_axis_xy[0])
        print(f"📐 Object principal axis angle in XY: {np.degrees(object_angle):.1f}°")
        
        # Generate grasps from 6 principal directions (±X, ±Y, ±Z)
        # For objects on a table, prefer top-down (Z-) grasps
        directions = [
            (np.array([0, 0, -1]), "Z-", 1.0),  # Top-down - HIGHEST priority
            (np.array([0, 0, 1]), "Z+", 0.3),   # Bottom-up - low priority
            (np.array([1, 0, 0]), "X+", 0.6),   # Side grasps - medium priority
            (np.array([-1, 0, 0]), "X-", 0.6),
            (np.array([0, 1, 0]), "Y+", 0.6),
            (np.array([0, -1, 0]), "Y-", 0.6),
        ]
        
        for approach_dir, label, base_quality in directions:
            # Estimate grasp width perpendicular to approach direction
            # Get the two dimensions perpendicular to approach
            perp_mask = np.abs(approach_dir) < 0.5
            perp_extents = bbox_extents[perp_mask]
            
            # Check if any perpendicular dimension fits in gripper
            if len(perp_extents) > 0:
                min_perp = np.min(perp_extents)
                max_perp = np.max(perp_extents)
                
                # Skip only if MINIMUM perpendicular extent is too large
                # (i.e., object is too thick to grasp from this direction)
                if min_perp > self.gripper_width:
                    print(f"  ⏭️ Skipping {label}: min width {min_perp:.3f}m > gripper {self.gripper_width}m")
                    continue
            
            # For top-down grasps, align with object orientation
            if label == "Z-":
                for rot_idx in range(num_orientations):
                    # Rotate perpendicular to object's long axis
                    angle = object_angle + (2 * np.pi * rot_idx) / num_orientations
                    
                    z_axis = approach_dir
                    x_axis = np.array([np.cos(angle), np.sin(angle), 0])
                    y_axis = np.cross(z_axis, x_axis)
                    
                    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
                    quat = self._rotation_matrix_to_quaternion(rotation_matrix)
                    
                    grasp_position = bbox_center - approach_dir * (np.max(bbox_extents) / 2 + self.standoff_distance)
                    
                    grasp_width = min_perp if len(perp_extents) > 0 else bbox_extents.min()
                    width_factor = 1.0 - (grasp_width / self.gripper_width) * 0.3
                    orientation_bonus = 0.1 if rot_idx == 0 else 0.0
                    quality = (base_quality + orientation_bonus) * width_factor
                    
                    grasp = GraspPose(position=grasp_position, orientation=quat, quality=quality)
                    self.candidate_grasps.append(grasp)
            else:
                # Other directions use standard rotation
                for rot_idx in range(num_orientations):
                    angle = (2 * np.pi * rot_idx) / num_orientations
                    
                    z_axis = approach_dir
                    if abs(z_axis[2]) < 0.9:
                        x_temp = np.array([0, 0, 1])
                    else:
                        x_temp = np.array([1, 0, 0])
                    
                    y_axis = np.cross(z_axis, x_temp)
                    y_axis = y_axis / np.linalg.norm(y_axis)
                    x_axis = np.cross(y_axis, z_axis)
                    
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    x_axis_rot = cos_a * x_axis + sin_a * y_axis
                    y_axis_rot = -sin_a * x_axis + cos_a * y_axis
                    
                    rotation_matrix = np.column_stack([x_axis_rot, y_axis_rot, z_axis])
                    quat = self._rotation_matrix_to_quaternion(rotation_matrix)
                    
                    grasp_position = bbox_center - approach_dir * (np.max(bbox_extents) / 2 + self.standoff_distance)
                    
                    grasp_width = min_perp if len(perp_extents) > 0 else bbox_extents.min()
                    width_factor = 1.0 - (grasp_width / self.gripper_width) * 0.3
                    quality = base_quality * width_factor
                    
                    grasp = GraspPose(position=grasp_position, orientation=quat, quality=quality)
                    self.candidate_grasps.append(grasp)
        
        print(f"✅ Generated {len(self.candidate_grasps)} bounding box grasps")
    
    def get_best_grasp(self, top_k: int = 1) -> Optional[GraspPose]:
        """
        Get the best grasp pose(s)
        
        Args:
            top_k: Number of top grasps to return (1 = best only)
            
        Returns:
            Best grasp pose or None if no grasps available
        """
        if not self.candidate_grasps:
            print("❌ No candidate grasps available")
            return None
        
        if top_k == 1:
            return self.candidate_grasps[0]
        else:
            return self.candidate_grasps[:top_k]
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 rotation matrix to quaternion [w, x, y, z]
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion [w, x, y, z]
        """
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    def _calculate_grasp_quality(
        self,
        normal1: np.ndarray,
        normal2: np.ndarray,
        grasp_width: float
    ) -> float:
        """
        Calculate grasp quality score
        
        Args:
            normal1: Normal at first contact point
            normal2: Normal at second contact point
            grasp_width: Distance between contact points
            
        Returns:
            Quality score [0, 1]
        """
        # Antipodal alignment: normals should be opposite
        alignment = -np.dot(normal1, normal2)  # 1 = perfect antipodal, -1 = same direction
        alignment_score = (alignment + 1) / 2.0  # Normalize to [0, 1]
        
        # Width score: prefer grasps closer to optimal width
        optimal_width = self.gripper_width * 0.6  # 60% of max width
        width_diff = abs(grasp_width - optimal_width)
        width_score = np.exp(-width_diff / optimal_width)
        
        # Combined quality
        quality = 0.7 * alignment_score + 0.3 * width_score
        
        return np.clip(quality, 0.0, 1.0)
