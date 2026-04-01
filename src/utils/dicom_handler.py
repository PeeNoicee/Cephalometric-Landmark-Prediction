"""
DICOM Image Handler for Cephalometric Landmark Detection

Converts DICOM (.dcm) files from cephalostat machines into 8-bit grayscale
images that match the PNG training data the model expects.

Key operations:
- Read DICOM pixel data and apply modality LUT / VOI LUT (windowing)
- Handle PhotometricInterpretation (MONOCHROME1 vs MONOCHROME2)
- Normalize high bit-depth (10-16 bit) to 8-bit
- Extract pixel spacing from DICOM metadata
- Return a PIL Image compatible with the existing GUI pipeline
"""
import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


def is_dicom_file(filepath: str) -> bool:
    """Check if a file is a DICOM file based on extension or magic bytes."""
    if filepath.lower().endswith(('.dcm', '.dicom')):
        return True
    # Check DICOM magic bytes (DICM at offset 128)
    try:
        with open(filepath, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False


def load_dicom_as_pil(filepath: str) -> Image.Image:
    """
    Load a DICOM file and convert it to an 8-bit grayscale PIL Image
    that visually matches the PNG cephalograms the model was trained on.

    Args:
        filepath: Path to the .dcm file

    Returns:
        PIL Image in RGB mode (consistent with existing GUI pipeline)

    Raises:
        ImportError: If pydicom is not installed
        ValueError: If the DICOM file cannot be read or has no pixel data
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError(
            "pydicom is required to load DICOM files.\n"
            "Install it with: pip install pydicom"
        )

    ds = pydicom.dcmread(filepath)

    if not hasattr(ds, 'pixel_array'):
        raise ValueError(f"DICOM file has no pixel data: {filepath}")

    # Step 1: Apply Modality LUT (Rescale Slope/Intercept) to get Hounsfield or display values
    pixel_array = apply_modality_lut(ds.pixel_array, ds)

    # Step 2: Apply VOI LUT (Window Center/Width) if present
    pixel_array = apply_voi_lut(pixel_array, ds, index=0)

    # Convert to float for safe manipulation
    pixel_array = pixel_array.astype(np.float64)

    # Step 3: Handle PhotometricInterpretation
    # MONOCHROME1: high values = dark (air), low values = bright (bone)
    # MONOCHROME2: high values = bright (bone), low values = dark (air)
    # Training PNGs are MONOCHROME2-style (bone = bright), so invert MONOCHROME1
    photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
    if photometric == 'MONOCHROME1':
        pixel_array = pixel_array.max() - pixel_array

    # Step 4: Handle multi-frame DICOM (take first frame)
    if pixel_array.ndim == 3:
        pixel_array = pixel_array[0]

    # Step 5: Normalize to 8-bit [0, 255]
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax - pmin > 0:
        pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
    else:
        pixel_array = np.zeros_like(pixel_array)

    img_8bit = pixel_array.astype(np.uint8)

    # Step 6: Convert to PIL Image in RGB (consistent with GUI's load_and_display_image)
    pil_image = Image.fromarray(img_8bit, mode='L').convert('RGB')

    return pil_image


def extract_dicom_pixel_spacing(filepath: str) -> float | None:
    """
    Extract pixel spacing (mm/pixel) from DICOM metadata.

    Checks multiple DICOM tags in priority order:
    1. ImagerPixelSpacing (0018,1164) — detector-level spacing, most accurate for ceph
    2. PixelSpacing (0028,0030) — image-level spacing
    3. NominalScannedPixelSpacing (0018,2010)

    Args:
        filepath: Path to the .dcm file

    Returns:
        Pixel spacing in mm/pixel, or None if not found
    """
    if not PYDICOM_AVAILABLE:
        return None

    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
    except Exception:
        return None

    # Priority 1: ImagerPixelSpacing (best for cephalometric X-rays)
    if hasattr(ds, 'ImagerPixelSpacing'):
        spacing = ds.ImagerPixelSpacing
        # Returns [row_spacing, col_spacing]; they are usually equal for ceph
        return float(spacing[0])

    # Priority 2: PixelSpacing
    if hasattr(ds, 'PixelSpacing'):
        spacing = ds.PixelSpacing
        return float(spacing[0])

    # Priority 3: NominalScannedPixelSpacing
    if hasattr(ds, 'NominalScannedPixelSpacing'):
        spacing = ds.NominalScannedPixelSpacing
        return float(spacing[0])

    return None


def get_dicom_metadata(filepath: str) -> dict:
    """
    Extract useful metadata from a DICOM cephalogram for display/reporting.

    Returns a dict with available fields (keys may be absent if not in file):
        - patient_name, patient_id, patient_dob, patient_sex
        - study_date, study_description
        - institution, manufacturer, station_name
        - pixel_spacing_mm
        - image_size (width, height)
        - bits_stored, bits_allocated
        - photometric_interpretation
    """
    if not PYDICOM_AVAILABLE:
        return {}

    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
    except Exception:
        return {}

    meta = {}

    # Patient info
    if hasattr(ds, 'PatientName'):
        meta['patient_name'] = str(ds.PatientName)
    if hasattr(ds, 'PatientID'):
        meta['patient_id'] = str(ds.PatientID)
    if hasattr(ds, 'PatientBirthDate'):
        meta['patient_dob'] = str(ds.PatientBirthDate)
    if hasattr(ds, 'PatientSex'):
        meta['patient_sex'] = str(ds.PatientSex)

    # Study info
    if hasattr(ds, 'StudyDate'):
        meta['study_date'] = str(ds.StudyDate)
    if hasattr(ds, 'StudyDescription'):
        meta['study_description'] = str(ds.StudyDescription)

    # Equipment info
    if hasattr(ds, 'InstitutionName'):
        meta['institution'] = str(ds.InstitutionName)
    if hasattr(ds, 'Manufacturer'):
        meta['manufacturer'] = str(ds.Manufacturer)
    if hasattr(ds, 'StationName'):
        meta['station_name'] = str(ds.StationName)

    # Pixel spacing
    spacing = extract_dicom_pixel_spacing(filepath)
    if spacing is not None:
        meta['pixel_spacing_mm'] = spacing

    # Image properties
    if hasattr(ds, 'Columns') and hasattr(ds, 'Rows'):
        meta['image_size'] = (int(ds.Columns), int(ds.Rows))
    if hasattr(ds, 'BitsStored'):
        meta['bits_stored'] = int(ds.BitsStored)
    if hasattr(ds, 'BitsAllocated'):
        meta['bits_allocated'] = int(ds.BitsAllocated)
    if hasattr(ds, 'PhotometricInterpretation'):
        meta['photometric_interpretation'] = str(ds.PhotometricInterpretation)

    return meta
