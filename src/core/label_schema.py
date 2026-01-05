"""
Unified Label Schema for Kepler-ECG Multi-Dataset Support

This module provides a hierarchical, extensible label system that:
1. Preserves fine-grained diagnostic details from each dataset
2. Enables cross-dataset comparison via unified categories
3. Supports multiple levels of granularity (superclass → subclass → specific)

Label Hierarchy:
    Level 0: Superclass (5 categories - PTB-XL compatible)
        - NORM: Normal ECG
        - MI: Myocardial Infarction
        - STTC: ST/T Changes
        - CD: Conduction Disturbance
        - HYP: Hypertrophy
    
    Level 1: Subclass (more specific categories)
        - e.g., MI → AMI (Acute MI), IMI (Inferior MI), etc.
    
    Level 2: Specific codes (original dataset codes)
        - SCP codes, SNOMED codes, etc.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
import re

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class Superclass(Enum):
    """
    Top-level diagnostic superclasses (PTB-XL compatible).
    These enable cross-dataset comparison.
    """
    NORM = "NORM"       # Normal ECG
    MI = "MI"           # Myocardial Infarction
    STTC = "STTC"       # ST/T Changes
    CD = "CD"           # Conduction Disturbance
    HYP = "HYP"         # Hypertrophy
    OTHER = "OTHER"     # Other/Unknown
    

class Subclass(Enum):
    """
    Mid-level diagnostic categories for finer granularity.
    """
    # Normal variants
    NORM_SINUS = "NORM_SINUS"           # Normal sinus rhythm
    NORM_VARIANT = "NORM_VARIANT"       # Normal variant
    
    # Myocardial Infarction subtypes
    MI_ACUTE = "MI_ACUTE"               # Acute MI
    MI_OLD = "MI_OLD"                   # Old/chronic MI
    MI_ANTERIOR = "MI_ANTERIOR"         # Anterior MI
    MI_INFERIOR = "MI_INFERIOR"         # Inferior MI
    MI_LATERAL = "MI_LATERAL"           # Lateral MI
    MI_POSTERIOR = "MI_POSTERIOR"       # Posterior MI
    MI_UNSPECIFIED = "MI_UNSPECIFIED"   # MI unspecified location
    
    # ST/T Changes subtypes
    STTC_ISCHEMIA = "STTC_ISCHEMIA"     # Ischemic changes
    STTC_INJURY = "STTC_INJURY"         # Injury pattern
    STTC_NONSPECIFIC = "STTC_NSP"       # Non-specific ST/T changes
    STTC_ELEVATION = "STTC_ELEV"        # ST elevation
    STTC_DEPRESSION = "STTC_DEP"        # ST depression
    STTC_TWAVE = "STTC_TWAVE"           # T wave abnormalities
    
    # Conduction Disturbance subtypes
    CD_LBBB = "CD_LBBB"                 # Left bundle branch block
    CD_RBBB = "CD_RBBB"                 # Right bundle branch block
    CD_LAFB = "CD_LAFB"                 # Left anterior fascicular block
    CD_LPFB = "CD_LPFB"                 # Left posterior fascicular block
    CD_AV_BLOCK = "CD_AVB"              # AV blocks
    CD_AV_BLOCK_1 = "CD_AVB1"           # First degree AV block
    CD_AV_BLOCK_2 = "CD_AVB2"           # Second degree AV block
    CD_AV_BLOCK_3 = "CD_AVB3"           # Third degree AV block
    CD_WPW = "CD_WPW"                   # Wolff-Parkinson-White
    CD_OTHER = "CD_OTHER"               # Other conduction issues
    
    # Hypertrophy subtypes
    HYP_LVH = "HYP_LVH"                 # Left ventricular hypertrophy
    HYP_RVH = "HYP_RVH"                 # Right ventricular hypertrophy
    HYP_LAE = "HYP_LAE"                 # Left atrial enlargement
    HYP_RAE = "HYP_RAE"                 # Right atrial enlargement
    HYP_BIVENTRICULAR = "HYP_BVH"       # Biventricular hypertrophy
    
    # Arrhythmias (often categorized under CD or OTHER)
    ARR_AFIB = "ARR_AFIB"               # Atrial fibrillation
    ARR_AFLUTTER = "ARR_AFL"            # Atrial flutter
    ARR_SINUS_TACHY = "ARR_STACHY"      # Sinus tachycardia
    ARR_SINUS_BRADY = "ARR_SBRADY"      # Sinus bradycardia
    ARR_PAC = "ARR_PAC"                 # Premature atrial contraction
    ARR_PVC = "ARR_PVC"                 # Premature ventricular contraction
    ARR_SVT = "ARR_SVT"                 # Supraventricular tachycardia
    ARR_VT = "ARR_VT"                   # Ventricular tachycardia
    
    # Other
    OTHER_UNCLASSIFIED = "OTHER_UNC"    # Unclassified
    OTHER_ARTIFACT = "OTHER_ART"        # Artifact/noise


@dataclass
class DiagnosticLabel:
    """
    Unified diagnostic label with hierarchical structure.
    """
    superclass: Superclass
    subclass: Optional[Subclass] = None
    original_code: Optional[str] = None
    original_description: Optional[str] = None
    confidence: float = 1.0
    source_dataset: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "superclass": self.superclass.value,
            "subclass": self.subclass.value if self.subclass else None,
            "original_code": self.original_code,
            "original_description": self.original_description,
            "confidence": self.confidence,
            "source_dataset": self.source_dataset,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosticLabel":
        """Create from dictionary."""
        return cls(
            superclass=Superclass(data["superclass"]),
            subclass=Subclass(data["subclass"]) if data.get("subclass") else None,
            original_code=data.get("original_code"),
            original_description=data.get("original_description"),
            confidence=data.get("confidence", 1.0),
            source_dataset=data.get("source_dataset"),
        )


# =============================================================================
# SCP Code Mapping (PTB-XL)
# =============================================================================

# Maps SCP codes to (Superclass, Subclass)
SCP_CODE_MAPPING: Dict[str, Tuple[Superclass, Optional[Subclass]]] = {
    # Normal
    "NORM": (Superclass.NORM, Subclass.NORM_SINUS),
    "SR": (Superclass.NORM, Subclass.NORM_SINUS),
    
    # Myocardial Infarction
    "IMI": (Superclass.MI, Subclass.MI_INFERIOR),
    "AMI": (Superclass.MI, Subclass.MI_ANTERIOR),
    "LMI": (Superclass.MI, Subclass.MI_LATERAL),
    "PMI": (Superclass.MI, Subclass.MI_POSTERIOR),
    "ASMI": (Superclass.MI, Subclass.MI_ANTERIOR),  # Anteroseptal MI
    "ILMI": (Superclass.MI, Subclass.MI_INFERIOR),  # Inferolateral MI
    "IPMI": (Superclass.MI, Subclass.MI_INFERIOR),  # Inferoposterior MI
    "IPLMI": (Superclass.MI, Subclass.MI_INFERIOR), # Inferoposterolateral MI
    "ALMI": (Superclass.MI, Subclass.MI_ANTERIOR),  # Anterolateral MI
    
    # ST/T Changes
    "NDT": (Superclass.STTC, Subclass.STTC_NONSPECIFIC),
    "NST_": (Superclass.STTC, Subclass.STTC_NONSPECIFIC),
    "DIG": (Superclass.STTC, Subclass.STTC_NONSPECIFIC),  # Digitalis effect
    "LNGQT": (Superclass.STTC, Subclass.STTC_TWAVE),      # Long QT
    "ISC_": (Superclass.STTC, Subclass.STTC_ISCHEMIA),
    "ISCA": (Superclass.STTC, Subclass.STTC_ISCHEMIA),    # Anterior ischemia
    "ISCI": (Superclass.STTC, Subclass.STTC_ISCHEMIA),    # Inferior ischemia
    "ISCLA": (Superclass.STTC, Subclass.STTC_ISCHEMIA),   # Lateral ischemia
    "ISCAS": (Superclass.STTC, Subclass.STTC_ISCHEMIA),   # Anteroseptal ischemia
    "ISCAL": (Superclass.STTC, Subclass.STTC_ISCHEMIA),   # Anterolateral ischemia
    "ISCIN": (Superclass.STTC, Subclass.STTC_ISCHEMIA),   # Inferolateral ischemia
    "ISCIL": (Superclass.STTC, Subclass.STTC_ISCHEMIA),   # Inferolateral ischemia
    "STD_": (Superclass.STTC, Subclass.STTC_DEPRESSION),
    "STE_": (Superclass.STTC, Subclass.STTC_ELEVATION),
    "INVT": (Superclass.STTC, Subclass.STTC_TWAVE),       # Inverted T waves
    "APTS": (Superclass.STTC, Subclass.STTC_TWAVE),       # Abnormal P/T waves
    "TAB_": (Superclass.STTC, Subclass.STTC_TWAVE),       # T wave abnormality
    "INJAL": (Superclass.STTC, Subclass.STTC_INJURY),     # Injury anterolateral
    "INJAS": (Superclass.STTC, Subclass.STTC_INJURY),     # Injury anteroseptal
    "INJIN": (Superclass.STTC, Subclass.STTC_INJURY),     # Injury inferior
    "INJLA": (Superclass.STTC, Subclass.STTC_INJURY),     # Injury lateral
    "INJIL": (Superclass.STTC, Subclass.STTC_INJURY),     # Injury inferolateral
    
    # Conduction Disturbances
    "CLBBB": (Superclass.CD, Subclass.CD_LBBB),    # Complete LBBB
    "CRBBB": (Superclass.CD, Subclass.CD_RBBB),    # Complete RBBB
    "ILBBB": (Superclass.CD, Subclass.CD_LBBB),    # Incomplete LBBB
    "IRBBB": (Superclass.CD, Subclass.CD_RBBB),    # Incomplete RBBB
    "LAFB": (Superclass.CD, Subclass.CD_LAFB),
    "LPFB": (Superclass.CD, Subclass.CD_LPFB),
    "1AVB": (Superclass.CD, Subclass.CD_AV_BLOCK_1),
    "2AVB": (Superclass.CD, Subclass.CD_AV_BLOCK_2),
    "3AVB": (Superclass.CD, Subclass.CD_AV_BLOCK_3),
    "AVB": (Superclass.CD, Subclass.CD_AV_BLOCK),
    "WPW": (Superclass.CD, Subclass.CD_WPW),
    "IVCD": (Superclass.CD, Subclass.CD_OTHER),    # Intraventricular conduction delay
    
    # Hypertrophy
    "LVH": (Superclass.HYP, Subclass.HYP_LVH),
    "RVH": (Superclass.HYP, Subclass.HYP_RVH),
    "LAO/LAE": (Superclass.HYP, Subclass.HYP_LAE),
    "RAO/RAE": (Superclass.HYP, Subclass.HYP_RAE),
    "SEHYP": (Superclass.HYP, Subclass.HYP_LVH),   # Septal hypertrophy
    "VCLVH": (Superclass.HYP, Subclass.HYP_LVH),   # Voltage criteria LVH
    
    # Arrhythmias
    "AFIB": (Superclass.CD, Subclass.ARR_AFIB),
    "AFLT": (Superclass.CD, Subclass.ARR_AFLUTTER),
    "STACH": (Superclass.CD, Subclass.ARR_SINUS_TACHY),
    "SBRAD": (Superclass.CD, Subclass.ARR_SINUS_BRADY),
    "SARRH": (Superclass.CD, Subclass.ARR_PAC),    # Sinus arrhythmia
    "SVTAC": (Superclass.CD, Subclass.ARR_SVT),
    "PSVT": (Superclass.CD, Subclass.ARR_SVT),
    "TRIGU": (Superclass.CD, Subclass.ARR_PVC),    # Trigeminy
    "BIGU": (Superclass.CD, Subclass.ARR_PVC),     # Bigeminy
    "PAC": (Superclass.CD, Subclass.ARR_PAC),
    "PVC": (Superclass.CD, Subclass.ARR_PVC),
}


# =============================================================================
# SNOMED-CT Code Mapping (Chapman, Challenge datasets)
# =============================================================================

# SNOMED-CT codes used in PhysioNet Challenge 2020/2021
SNOMED_CODE_MAPPING: Dict[str, Tuple[Superclass, Optional[Subclass], str]] = {
    # Normal
    "426783006": (Superclass.NORM, Subclass.NORM_SINUS, "Sinus rhythm"),
    "426177001": (Superclass.NORM, Subclass.NORM_SINUS, "Sinus bradycardia"),  # Could also be ARR
    
    # Atrial Fibrillation / Flutter
    "164889003": (Superclass.CD, Subclass.ARR_AFIB, "Atrial fibrillation"),
    "164890007": (Superclass.CD, Subclass.ARR_AFLUTTER, "Atrial flutter"),
    
    # Bundle Branch Blocks
    "713427006": (Superclass.CD, Subclass.CD_LBBB, "Complete LBBB"),
    "713426002": (Superclass.CD, Subclass.CD_RBBB, "Complete RBBB"),
    "445118002": (Superclass.CD, Subclass.CD_LAFB, "Left anterior fascicular block"),
    "445211001": (Superclass.CD, Subclass.CD_LPFB, "Left posterior fascicular block"),
    
    # AV Blocks
    "270492004": (Superclass.CD, Subclass.CD_AV_BLOCK_1, "First degree AV block"),
    "195042002": (Superclass.CD, Subclass.CD_AV_BLOCK_2, "Second degree AV block, Mobitz I"),
    "426995002": (Superclass.CD, Subclass.CD_AV_BLOCK_2, "Second degree AV block, Mobitz II"),
    "27885002": (Superclass.CD, Subclass.CD_AV_BLOCK_3, "Third degree AV block"),
    
    # Premature beats
    "284470004": (Superclass.CD, Subclass.ARR_PAC, "Premature atrial contraction"),
    "427172004": (Superclass.CD, Subclass.ARR_PVC, "Premature ventricular contraction"),
    
    # Tachycardias
    "427084000": (Superclass.CD, Subclass.ARR_SINUS_TACHY, "Sinus tachycardia"),
    "426761007": (Superclass.CD, Subclass.ARR_SVT, "Supraventricular tachycardia"),
    "713422000": (Superclass.CD, Subclass.ARR_VT, "Ventricular tachycardia"),
    
    # ST/T Changes
    "428750005": (Superclass.STTC, Subclass.STTC_NONSPECIFIC, "Nonspecific ST abnormality"),
    "164930006": (Superclass.STTC, Subclass.STTC_ELEVATION, "ST elevation"),
    "164931005": (Superclass.STTC, Subclass.STTC_DEPRESSION, "ST depression"),
    "164934002": (Superclass.STTC, Subclass.STTC_TWAVE, "T wave abnormal"),
    "59931005": (Superclass.STTC, Subclass.STTC_TWAVE, "T wave inversion"),
    "164947007": (Superclass.STTC, Subclass.STTC_TWAVE, "Prolonged QT"),
    
    # Myocardial Infarction
    "164865005": (Superclass.MI, Subclass.MI_UNSPECIFIED, "Myocardial infarction"),
    "54329005": (Superclass.MI, Subclass.MI_ACUTE, "Acute MI"),
    "73795002": (Superclass.MI, Subclass.MI_OLD, "Old MI"),
    "164861001": (Superclass.MI, Subclass.MI_ANTERIOR, "Anterior MI"),
    "164867002": (Superclass.MI, Subclass.MI_INFERIOR, "Inferior MI"),
    "164868007": (Superclass.MI, Subclass.MI_LATERAL, "Lateral MI"),
    
    # Hypertrophy
    "164873001": (Superclass.HYP, Subclass.HYP_LVH, "Left ventricular hypertrophy"),
    "89792004": (Superclass.HYP, Subclass.HYP_RVH, "Right ventricular hypertrophy"),
    "67741000119109": (Superclass.HYP, Subclass.HYP_LAE, "Left atrial enlargement"),
    "446358003": (Superclass.HYP, Subclass.HYP_RAE, "Right atrial enlargement"),
    
    # Pacing
    "10370003": (Superclass.OTHER, None, "Paced rhythm"),
    "251120003": (Superclass.OTHER, None, "Ventricular pacing"),
    
    # Low voltage
    "251146004": (Superclass.OTHER, None, "Low QRS voltages"),
    
    # Q waves
    "164917005": (Superclass.MI, Subclass.MI_OLD, "Q wave abnormal"),
}


# =============================================================================
# Label Mapper Class
# =============================================================================

class LabelMapper:
    """
    Maps dataset-specific labels to unified schema.
    
    Usage:
        mapper = LabelMapper("ptb-xl")
        labels = mapper.map_scp_codes({"NORM": 100.0, "SR": 80.0})
        print(labels[0].superclass)  # Superclass.NORM
    """
    
    def __init__(self, dataset_name: str):
        """
        Initialize mapper for a specific dataset.
        
        Args:
            dataset_name: Name of the source dataset
        """
        self.dataset_name = dataset_name.lower()
        self._unknown_codes: Set[str] = set()
    
    def map_scp_codes(
        self, 
        scp_codes: Dict[str, float],
        threshold: float = 0.0
    ) -> List[DiagnosticLabel]:
        """
        Map PTB-XL style SCP codes to unified labels.
        
        Args:
            scp_codes: Dictionary of {code: confidence}
            threshold: Minimum confidence threshold
            
        Returns:
            List of DiagnosticLabel objects
        """
        labels = []
        
        for code, confidence in scp_codes.items():
            if confidence < threshold:
                continue
                
            code_upper = code.upper().strip()
            
            if code_upper in SCP_CODE_MAPPING:
                superclass, subclass = SCP_CODE_MAPPING[code_upper]
                labels.append(DiagnosticLabel(
                    superclass=superclass,
                    subclass=subclass,
                    original_code=code,
                    confidence=confidence / 100.0 if confidence > 1 else confidence,
                    source_dataset=self.dataset_name,
                ))
            else:
                # Try partial match
                matched = False
                for pattern, (superclass, subclass) in SCP_CODE_MAPPING.items():
                    if code_upper.startswith(pattern.rstrip('_')):
                        labels.append(DiagnosticLabel(
                            superclass=superclass,
                            subclass=subclass,
                            original_code=code,
                            confidence=confidence / 100.0 if confidence > 1 else confidence,
                            source_dataset=self.dataset_name,
                        ))
                        matched = True
                        break
                
                if not matched:
                    self._unknown_codes.add(code)
                    labels.append(DiagnosticLabel(
                        superclass=Superclass.OTHER,
                        subclass=Subclass.OTHER_UNCLASSIFIED,
                        original_code=code,
                        confidence=confidence / 100.0 if confidence > 1 else confidence,
                        source_dataset=self.dataset_name,
                    ))
        
        return labels
    
    def map_snomed_codes(
        self, 
        snomed_codes: List[str]
    ) -> List[DiagnosticLabel]:
        """
        Map SNOMED-CT codes to unified labels.
        
        Args:
            snomed_codes: List of SNOMED-CT code strings
            
        Returns:
            List of DiagnosticLabel objects
        """
        labels = []
        
        for code in snomed_codes:
            code_str = str(code).strip()
            
            if code_str in SNOMED_CODE_MAPPING:
                superclass, subclass, description = SNOMED_CODE_MAPPING[code_str]
                labels.append(DiagnosticLabel(
                    superclass=superclass,
                    subclass=subclass,
                    original_code=code_str,
                    original_description=description,
                    confidence=1.0,
                    source_dataset=self.dataset_name,
                ))
            else:
                self._unknown_codes.add(code_str)
                labels.append(DiagnosticLabel(
                    superclass=Superclass.OTHER,
                    subclass=Subclass.OTHER_UNCLASSIFIED,
                    original_code=code_str,
                    confidence=1.0,
                    source_dataset=self.dataset_name,
                ))
        
        return labels
    
    def get_primary_superclass(
        self, 
        labels: List[DiagnosticLabel]
    ) -> Superclass:
        """
        Determine the primary superclass from a list of labels.
        Priority: MI > STTC > CD > HYP > OTHER > NORM
        
        Args:
            labels: List of diagnostic labels
            
        Returns:
            Primary Superclass
        """
        if not labels:
            return Superclass.OTHER
        
        # Priority order (higher priority first)
        priority = {
            Superclass.MI: 5,
            Superclass.STTC: 4,
            Superclass.CD: 3,
            Superclass.HYP: 2,
            Superclass.OTHER: 1,
            Superclass.NORM: 0,
        }
        
        # Find highest priority with highest confidence
        best_label = max(
            labels, 
            key=lambda l: (priority.get(l.superclass, 0), l.confidence)
        )
        return best_label.superclass
    
    def get_superclass_vector(
        self, 
        labels: List[DiagnosticLabel]
    ) -> Dict[str, int]:
        """
        Create binary vector of superclasses present.
        
        Args:
            labels: List of diagnostic labels
            
        Returns:
            Dictionary of {superclass_name: 0 or 1}
        """
        vector = {sc.value: 0 for sc in Superclass}
        for label in labels:
            vector[label.superclass.value] = 1
        return vector
    
    @property
    def unknown_codes(self) -> Set[str]:
        """Get set of codes that couldn't be mapped."""
        return self._unknown_codes.copy()
    
    def report_unknown_codes(self) -> str:
        """Generate report of unknown codes."""
        if not self._unknown_codes:
            return "All codes successfully mapped."
        return f"Unknown codes ({len(self._unknown_codes)}): {sorted(self._unknown_codes)}"


# =============================================================================
# Utility Functions
# =============================================================================

def parse_header_labels(header_path: str) -> List[str]:
    """
    Parse diagnostic labels from WFDB header file comments.
    Used for Chapman, CPSC-2018, Georgia datasets.
    
    Args:
        header_path: Path to .hea file
        
    Returns:
        List of SNOMED-CT codes found
    """
    codes = []
    
    try:
        with open(header_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Look for #Dx: line (PhysioNet Challenge format)
                if line.startswith('#Dx:'):
                    # Format: #Dx: code1,code2,code3
                    dx_part = line.replace('#Dx:', '').strip()
                    codes = [c.strip() for c in dx_part.split(',') if c.strip()]
                    break
    except Exception as e:
        logger.warning(f"Error parsing header {header_path}: {e}")
    
    return codes


def create_label_dataframe_columns() -> Dict[str, type]:
    """
    Get column definitions for a DataFrame with label information.
    
    Returns:
        Dictionary of {column_name: dtype}
    """
    columns = {
        # Primary classification
        'primary_superclass': str,
        # Binary superclass flags
        'label_NORM': int,
        'label_MI': int,
        'label_STTC': int,
        'label_CD': int,
        'label_HYP': int,
        'label_OTHER': int,
        # Original codes (as string list)
        'original_codes': str,
        # Confidence scores
        'max_confidence': float,
    }
    return columns


# =============================================================================
# CLI Support
# =============================================================================

if __name__ == "__main__":
    # Demo usage
    print("Label Schema Demo")
    print("=" * 50)
    
    # PTB-XL example
    mapper = LabelMapper("ptb-xl")
    scp_codes = {"NORM": 100.0, "SR": 80.0, "LBBB": 50.0}
    labels = mapper.map_scp_codes(scp_codes)
    
    print("\nPTB-XL SCP codes:", scp_codes)
    print("Mapped labels:")
    for label in labels:
        print(f"  {label.superclass.value} / {label.subclass.value if label.subclass else 'N/A'}")
    
    print(f"\nPrimary superclass: {mapper.get_primary_superclass(labels).value}")
    print(f"Superclass vector: {mapper.get_superclass_vector(labels)}")
    
    # SNOMED example
    print("\n" + "=" * 50)
    mapper2 = LabelMapper("chapman")
    snomed_codes = ["426783006", "713427006"]
    labels2 = mapper2.map_snomed_codes(snomed_codes)
    
    print("Chapman SNOMED codes:", snomed_codes)
    print("Mapped labels:")
    for label in labels2:
        print(f"  {label.superclass.value} / {label.subclass.value if label.subclass else 'N/A'}")
        print(f"    Original: {label.original_description}")
