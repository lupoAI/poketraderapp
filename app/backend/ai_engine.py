
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import pickle
import faiss

_DEFAULT_AVG_FILENAME = "avg_embedding.npy"

# Paths - ADJUST THESE TO MATCH YOUR ACTUAL PATHS FROM THE REFERENCE
# Assumes app/backend/ai_engine.py -> ../../poketrader/storage/
REPO_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = REPO_ROOT / "poketrader" / "storage"
YOLO_WEIGHTS = STORAGE_DIR / "model_weights" / "pokemon-yolo11n-seg-v3.pt"
INDEX_DIR = STORAGE_DIR / "indices" / "model=clip__q=high"

_DEFAULT_AVG_FILENAME = "avg_embedding.npy"

class AIEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading YOLO model on {self.device}...")
        self.yolo = YOLO(str(YOLO_WEIGHTS)).to(self.device)
        
        print(f"Loading Index bundle from {INDEX_DIR}...")
        # Load centered_normalized variant as it's the most accurate
        self.index = faiss.read_index(str(INDEX_DIR / "pokemon_cards_centered_norm.index"))
        self.card_names = np.load(INDEX_DIR / "card_names.npy", allow_pickle=True)
        self.avg_embedding = np.load(INDEX_DIR / _DEFAULT_AVG_FILENAME).astype(np.float32).reshape(1, -1)
            
        import open_clip
        print(f"Loading CLIP model (ViT-B-32, laion2b_s34b_b79k) on {self.device}...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", image_resize_mode="longest"
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

    def _get_birdseye_view(self, image, corners):
        # corners: [top-left, top-right, bottom-right, bottom-left]
        width, height = 400, 550
        dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
        M = cv2.getPerspectiveTransform(np.array(corners, dtype="float32"), dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))
        return warped

    def _order_points(self, pts):
        pts = pts.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.stack([tl, tr, br, bl], axis=0)

    def _try_approx_to_4(self, contour):
        peri = cv2.arcLength(contour, True)
        for frac in (0.002, 0.005, 0.01, 0.02, 0.04):
            approx = cv2.approxPolyDP(contour, frac * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2)
        return None

    def _get_quad_from_mask(self, mask_xy):
        pts = np.asarray(mask_xy, dtype=np.float32)
        contour = pts.reshape(-1, 1, 2).astype(np.float32)
        
        approx = self._try_approx_to_4(contour)
        if approx is not None:
            return self._order_points(approx)
            
        hull = cv2.convexHull(contour)
        approx = self._try_approx_to_4(hull)
        if approx is not None:
            return self._order_points(approx)
            
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        return self._order_points(box)

    def detect_and_identify(self, image_bytes, is_cropped=False):
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
            
        # NOTE: If we move cropping to client, we can bypass this detection phase.
        if is_cropped:
            # Bypass detection, use frame as warped result directly
            print("Processing pre-cropped image...")
            return self._identify_cropped(frame)

        # 1. Detect
        print("Running YOLO detection...")
        results = self.yolo.predict(frame, verbose=False, conf=0.6)
        
        if not results or not results[0].masks:
            print("No masks detected in frame.")
            return []
            
        r = results[0]
        h, w = frame.shape[:2]
        matches = []
        
        for i, mask in enumerate(r.masks.xy):
            if len(mask) < 4: continue
            
            # YOLO confidence for this box
            yolo_conf = float(r.boxes.conf[i])
            
            # Extract center of mask for UI positioning (relative)
            poly = mask.astype(np.float32)
            moments = cv2.moments(poly)
            if moments["m00"] != 0:
                cx = (moments["m10"] / moments["m00"]) / w
                cy = (moments["m01"] / moments["m00"]) / h
            else:
                # Fallback to bbox center
                box = r.boxes.xyxyn[i].detach().cpu().numpy()
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2

            # Improved quad extraction
            quad = self._get_quad_from_mask(mask)
            if quad is not None:
                area = abs(cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.float32)))
                if area < 1000: continue
                
                warped = self._get_birdseye_view(frame, quad)
                
                # Embed with CLIP (RAW)
                pil_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                image_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.clip_model.encode_image(image_tensor).cpu().numpy().astype(np.float32)
                    
                # Center and Normalize
                embedding = embedding - self.avg_embedding
                faiss.normalize_L2(embedding)
                
                # Search Top 5
                D, I = self.index.search(embedding, 5)
                top_matches = []
                for j in range(len(I[0])):
                    idx = I[0][j]
                    if idx < 0: continue
                    top_matches.append({
                        "card_id": str(self.card_names[idx]),
                        "confidence": float(D[0][j])
                    })
                
                if top_matches:
                    matches.append({
                        "top_matches": top_matches,
                        "yolo_conf": yolo_conf,
                        "center": {"x": float(cx), "y": float(cy)}
                    })
                    print(f"Match found: {top_matches[0]['card_id']} with CLIP conf {top_matches[0]['confidence']}")
        
        return matches
                    
        print("No card detected in frame.")
        return None

    def _identify_cropped(self, warped):
        # Embed with CLIP
        pil_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        image_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.clip_model.encode_image(image_tensor).cpu().numpy().astype(np.float32)
            
        # Center and Normalize
        embedding = embedding - self.avg_embedding
        faiss.normalize_L2(embedding)
        
        # Search
        D, I = self.index.search(embedding, 1)
        card_idx = I[0][0]
        confidence = D[0][0]
        card_id = self.card_names[card_idx]
        
        return {
            "card_id": str(card_id),
            "confidence": float(confidence),
        }
