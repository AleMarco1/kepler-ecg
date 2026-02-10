/**
 * Marconi QTc Formula - JavaScript Implementation
 * ================================================
 * 
 * The Marconi formula for heart rate-corrected QT interval.
 * 
 * Formula: QTc = QT + 125/RR - 158
 * 
 * Where:
 *   QTc = Corrected QT interval (ms)
 *   QT  = Measured QT interval (ms)
 *   RR  = RR interval (seconds)
 * 
 * Reference:
 *   Marconi A. (2026). A Novel QTc Correction Formula Derived from 
 *   Symbolic Regression on 1.2 Million ECGs.
 * 
 * @author Alessandro Marconi
 * @version 1.0.0
 * @license MIT
 */

// Formula coefficients
const K = 125;  // Correction coefficient (ms·s)
const C = 158;  // Calibration constant (ms)

/**
 * Calculate QTc using the Marconi formula
 * 
 * @param {number} qtMs - QT interval in milliseconds
 * @param {number} rrSec - RR interval in seconds
 * @param {boolean} [validate=true] - Whether to validate inputs
 * @returns {number} QTc in milliseconds
 * @throws {Error} If inputs are outside valid ranges
 * 
 * @example
 * calculateQTcMarconi(380, 0.8);  // Returns 378.25
 */
function calculateQTcMarconi(qtMs, rrSec, validate = true) {
    if (validate) {
        if (qtMs < 200 || qtMs > 600) {
            throw new Error('QT must be between 200 and 600 ms');
        }
        if (rrSec < 0.4 || rrSec > 2.0) {
            throw new Error('RR must be between 0.4 and 2.0 seconds');
        }
    }
    
    return qtMs + K / rrSec - C;
}

/**
 * Calculate QTc using heart rate instead of RR interval
 * 
 * @param {number} qtMs - QT interval in milliseconds
 * @param {number} hrBpm - Heart rate in beats per minute
 * @param {boolean} [validate=true] - Whether to validate inputs
 * @returns {number} QTc in milliseconds
 * 
 * @example
 * calculateQTcMarconiFromHR(380, 75);  // Returns 378.25
 */
function calculateQTcMarconiFromHR(qtMs, hrBpm, validate = true) {
    const rrSec = 60 / hrBpm;
    return calculateQTcMarconi(qtMs, rrSec, validate);
}

/**
 * Classify QTc into clinical categories
 * 
 * Uses AHA/ACC sex-specific thresholds.
 * 
 * @param {number} qtcMs - QTc in milliseconds
 * @param {string} [sex='M'] - 'M' for male, 'F' for female
 * @returns {string} Classification: 'normal', 'borderline', 'prolonged', or 'high_risk'
 * 
 * @example
 * classifyQTc(420, 'M');  // Returns 'normal'
 * classifyQTc(465, 'F');  // Returns 'borderline'
 */
function classifyQTc(qtcMs, sex = 'M') {
    const thresholdNormal = sex.toUpperCase() === 'M' ? 450 : 460;
    
    if (qtcMs < thresholdNormal) return 'normal';
    if (qtcMs < 470) return 'borderline';
    if (qtcMs < 500) return 'prolonged';
    return 'high_risk';
}

/**
 * Compare Marconi and Bazett QTc values
 * 
 * @param {number} qtMs - QT interval in milliseconds
 * @param {number} rrSec - RR interval in seconds
 * @returns {Object} Object with marconi, bazett, and difference values
 * 
 * @example
 * compareWithBazett(380, 0.8);
 * // Returns { marconi: 378.25, bazett: 424.85, difference: -46.6 }
 */
function compareWithBazett(qtMs, rrSec) {
    const qtcMarconi = qtMs + K / rrSec - C;
    const qtcBazett = qtMs / Math.sqrt(rrSec);
    
    return {
        marconi: qtcMarconi,
        bazett: qtcBazett,
        difference: qtcMarconi - qtcBazett
    };
}

/**
 * Calculate all common QTc formulas at once
 * 
 * @param {number} qtMs - QT interval in milliseconds
 * @param {number} rrSec - RR interval in seconds
 * @returns {Object} Object with all QTc values
 */
function calculateAllFormulas(qtMs, rrSec) {
    const hrBpm = 60 / rrSec;
    
    return {
        marconi: qtMs + K / rrSec - C,
        bazett: qtMs / Math.sqrt(rrSec),
        fridericia: qtMs / Math.cbrt(rrSec),
        framingham: qtMs + 154 * (1 - rrSec),
        hodges: qtMs + 1.75 * (hrBpm - 60)
    };
}

// Convenience functions for other common formulas

/**
 * Bazett formula: QTc = QT / sqrt(RR)
 */
function calculateQTcBazett(qtMs, rrSec) {
    return qtMs / Math.sqrt(rrSec);
}

/**
 * Fridericia formula: QTc = QT / cbrt(RR)
 */
function calculateQTcFridericia(qtMs, rrSec) {
    return qtMs / Math.cbrt(rrSec);
}

/**
 * Framingham formula: QTc = QT + 154(1-RR)
 */
function calculateQTcFramingham(qtMs, rrSec) {
    return qtMs + 154 * (1 - rrSec);
}

/**
 * Hodges formula: QTc = QT + 1.75(HR-60)
 */
function calculateQTcHodges(qtMs, hrBpm) {
    return qtMs + 1.75 * (hrBpm - 60);
}

// Export for Node.js / CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        calculateQTcMarconi,
        calculateQTcMarconiFromHR,
        classifyQTc,
        compareWithBazett,
        calculateAllFormulas,
        calculateQTcBazett,
        calculateQTcFridericia,
        calculateQTcFramingham,
        calculateQTcHodges,
        K,
        C
    };
}

// Export for ES6 modules
if (typeof exports !== 'undefined') {
    exports.calculateQTcMarconi = calculateQTcMarconi;
    exports.calculateQTcMarconiFromHR = calculateQTcMarconiFromHR;
    exports.classifyQTc = classifyQTc;
    exports.compareWithBazett = compareWithBazett;
    exports.calculateAllFormulas = calculateAllFormulas;
}

// Example usage (runs when executed directly in Node.js)
if (typeof require !== 'undefined' && require.main === module) {
    console.log("Marconi QTc Formula Calculator");
    console.log("=".repeat(40));
    
    const qt = 380;  // ms
    const rr = 0.8;  // seconds (75 bpm)
    
    const qtc = calculateQTcMarconi(qt, rr);
    const classification = classifyQTc(qtc, 'M');
    const comparison = compareWithBazett(qt, rr);
    
    console.log(`\nInput: QT = ${qt} ms, RR = ${rr} s (HR = ${60/rr} bpm)`);
    console.log(`\nMarconi QTc:  ${qtc.toFixed(1)} ms (${classification})`);
    console.log(`Bazett QTc:   ${comparison.bazett.toFixed(1)} ms`);
    console.log(`Difference:   ${comparison.difference.toFixed(1)} ms`);
    
    console.log("\n" + "=".repeat(40));
    console.log("All formulas comparison:");
    const all = calculateAllFormulas(qt, rr);
    Object.entries(all).forEach(([name, value]) => {
        console.log(`  ${name.padEnd(12)}: ${value.toFixed(1)} ms`);
    });
}
