from enum import Enum

class PaperSelection(Enum):
    AGNOLUCCI = "Agnolucci_WACV_2024"
    HENDRYCKS = "Hendrycks_ICLR_2019"
    LIU = "Liu_IJCV_2024"

PAPER_NAME_MAP = {
    PaperSelection.AGNOLUCCI: "Agnolucci_WACV_2024",
    PaperSelection.HENDRYCKS: "Hendrycks_ICLR_2019",
    PaperSelection.LIU: "Liu_IJCV_2024",
}