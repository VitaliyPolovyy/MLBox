## Peanuts Service – Segmentation Comparison: Specification

### 1. Goal

Implement a new separated segmentation approach while keeping the existing full-image segmentation approach working exactly as before. Add visual comparison in Excel output.

**Key Requirements**:
- ✅ Old approach (full image segmentation) continues to work unchanged
- ✅ All existing Excel sheets remain unchanged
- ✅ All measurements and results use old approach (backward compatible)
- ✅ Add ONE new Excel sheet showing both segmentations on one image for visual comparison

---

### 2. Implementation Steps

1. **Add New Model Configuration**
   - Add environment variables for separated segmentation model
   - Load model conditionally (only if env vars set)

2. **Extend Data Structure**
   - Add to `OnePeanutProcessingResult`:
     - `mask_separated: Optional[np.ndarray] = None` - separated segmentation mask (full image coordinates)
     - `contour_separated: Optional[np.ndarray] = None` - contour from separated mask
     - `ellipse_separated: Optional[Ellipse] = None` - ellipse fitted from separated contour
   - All fields optional, default `None` for backward compatibility

3. **Create New Function for Separated Segmentation**
   - New function to process cropped peanuts with separated model
   - Input: cropped peanut images, separated model
   - Output: masks in full image coordinates
   - Handle coordinate translation from cropped to full image

4. **Integrate into Processing Pipeline**
   - Add separated segmentation step (conditional)
   - Store results without affecting existing processing

5. **Add Comparison Visualization**
   - Create comparison image with both segmentations
   - Add comparison sheet to Excel output

6. **Testing**
   - Test backward compatibility (without separated model)
   - Test with separated model
   - Validate Excel output
