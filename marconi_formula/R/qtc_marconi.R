#' Marconi QTc Formula - R Implementation
#' ========================================
#'
#' The Marconi formula for heart rate-corrected QT interval.
#'
#' Formula: QTc = QT + 125/RR - 158
#'
#' Where:
#'   QTc = Corrected QT interval (ms)
#'   QT  = Measured QT interval (ms)
#'   RR  = RR interval (seconds)
#'
#' Reference:
#'   Marconi A. (2026). A Novel QTc Correction Formula Derived from 
#'   Symbolic Regression on 1.2 Million ECGs.
#'
#' @author Alessandro Marconi
#' @version 1.0.0

# Formula coefficients
K <- 125  # Correction coefficient (ms·s)
C <- 158  # Calibration constant (ms)


#' Calculate QTc using the Marconi formula
#'
#' @param qt_ms QT interval in milliseconds (numeric or vector)
#' @param rr_sec RR interval in seconds (numeric or vector)
#' @param validate Whether to validate input ranges (default: TRUE)
#' @return QTc in milliseconds
#' @export
#' @examples
#' calculate_qtc_marconi(380, 0.8)  # Returns 378.25
#' calculate_qtc_marconi(c(380, 400), c(0.8, 1.0))  # Vectorized
calculate_qtc_marconi <- function(qt_ms, rr_sec, validate = TRUE) {
  if (validate) {
    if (any(qt_ms < 200 | qt_ms > 600, na.rm = TRUE)) {
      stop("QT must be between 200 and 600 ms")
    }
    if (any(rr_sec < 0.4 | rr_sec > 2.0, na.rm = TRUE)) {
      stop("RR must be between 0.4 and 2.0 seconds")
    }
  }
  
  qt_ms + K / rr_sec - C
}


#' Calculate QTc from heart rate
#'
#' @param qt_ms QT interval in milliseconds
#' @param hr_bpm Heart rate in beats per minute
#' @param validate Whether to validate input ranges (default: TRUE)
#' @return QTc in milliseconds
#' @export
#' @examples
#' calculate_qtc_marconi_from_hr(380, 75)  # Returns 378.25
calculate_qtc_marconi_from_hr <- function(qt_ms, hr_bpm, validate = TRUE) {
  rr_sec <- 60 / hr_bpm
  calculate_qtc_marconi(qt_ms, rr_sec, validate)
}


#' Classify QTc into clinical categories
#'
#' Uses AHA/ACC sex-specific thresholds.
#'
#' @param qtc_ms QTc in milliseconds
#' @param sex 'M' for male, 'F' for female (default: 'M')
#' @return Character string with classification
#' @export
#' @examples
#' classify_qtc(420, "M")  # Returns "normal"
#' classify_qtc(465, "F")  # Returns "borderline"
classify_qtc <- function(qtc_ms, sex = "M") {
  threshold_normal <- ifelse(toupper(sex) == "M", 450, 460)
  
  dplyr::case_when(
    qtc_ms < threshold_normal ~ "normal",
    qtc_ms < 470 ~ "borderline",
    qtc_ms < 500 ~ "prolonged",
    TRUE ~ "high_risk"
  )
}


#' Classify QTc (base R version, no dplyr dependency)
#'
#' @param qtc_ms QTc in milliseconds
#' @param sex 'M' for male, 'F' for female (default: 'M')
#' @return Character string with classification
#' @export
classify_qtc_base <- function(qtc_ms, sex = "M") {
  threshold_normal <- ifelse(toupper(sex) == "M", 450, 460)
  
  ifelse(qtc_ms < threshold_normal, "normal",
         ifelse(qtc_ms < 470, "borderline",
                ifelse(qtc_ms < 500, "prolonged", "high_risk")))
}


#' Compare Marconi with Bazett formula
#'
#' @param qt_ms QT interval in milliseconds
#' @param rr_sec RR interval in seconds
#' @return Data frame with marconi, bazett, and difference columns
#' @export
compare_with_bazett <- function(qt_ms, rr_sec) {
  qtc_marconi <- qt_ms + K / rr_sec - C
  qtc_bazett <- qt_ms / sqrt(rr_sec)
  
  data.frame(
    marconi = qtc_marconi,
    bazett = qtc_bazett,
    difference = qtc_marconi - qtc_bazett
  )
}


# Convenience functions for other common formulas

#' Bazett formula: QTc = QT / sqrt(RR)
#' @export
calculate_qtc_bazett <- function(qt_ms, rr_sec) {
  qt_ms / sqrt(rr_sec)
}

#' Fridericia formula: QTc = QT / cbrt(RR)
#' @export
calculate_qtc_fridericia <- function(qt_ms, rr_sec) {
  qt_ms / (rr_sec^(1/3))
}

#' Framingham formula: QTc = QT + 154(1-RR)
#' @export
calculate_qtc_framingham <- function(qt_ms, rr_sec) {
  qt_ms + 154 * (1 - rr_sec)
}

#' Hodges formula: QTc = QT + 1.75(HR-60)
#' @export
calculate_qtc_hodges <- function(qt_ms, hr_bpm) {
  qt_ms + 1.75 * (hr_bpm - 60)
}


# Example usage when sourced directly
if (interactive()) {
  cat("Marconi QTc Formula Calculator\n")
  cat(strrep("=", 40), "\n")
  
  # Single value example
  qt <- 380  # ms
  rr <- 0.8  # seconds (75 bpm)
  
  qtc <- calculate_qtc_marconi(qt, rr)
  comparison <- compare_with_bazett(qt, rr)
  
  cat(sprintf("\nInput: QT = %d ms, RR = %.1f s (HR = %.0f bpm)\n", 
              qt, rr, 60/rr))
  cat(sprintf("\nMarconi QTc:  %.1f ms (%s)\n", 
              qtc, classify_qtc_base(qtc, "M")))
  cat(sprintf("Bazett QTc:   %.1f ms\n", comparison$bazett))
  cat(sprintf("Difference:   %.1f ms\n", comparison$difference))
}
