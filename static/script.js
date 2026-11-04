
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const result = document.getElementById("result");
let drawing = false;
// White Canvas
function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(
        0,
        0,
        canvas.width,
        canvas.height
    );
    result.innerHTML = `
        <div class="placeholder">
            <span>?</span>
            <p>Your predicted digit will appear here.</p>
        </div>
    `;
}
// Drawing Settings
ctx.lineWidth = 22;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "black";
// Get Mouse Position
function getPosition(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left) * canvas.width / rect.width,
        y: (e.clientY - rect.top) * canvas.height / rect.height
    };
}
// Start Drawing
canvas.addEventListener("pointerdown", (e) => {
    drawing = true;
    const pos = getPosition(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    ctx.lineTo(pos.x + 0.1, pos.y + 0.1);
    ctx.stroke();
    canvas.setPointerCapture(e.pointerId);

});
// Draw Digit
canvas.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const pos = getPosition(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
});
// Stop Drawing
function stopDrawing() {
    drawing = false;
    ctx.closePath();
}
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);
// Clear Button
clearBtn.addEventListener("click", clearCanvas);
// Predict Button
predictBtn.addEventListener("click", async () => {
    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";
    result.innerHTML = `
        <p class="placeholder">Please wait...</p>
    `;
    try {
        // Convert Canvas into Image
        const image = canvas.toDataURL("image/png");
        // Send Image to Flask Backend
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: image
            })
        });
        const data = await response.json();
        // Check Error
        if (!response.ok || data.error) {
            throw new Error(
                data.error || "Prediction failed"
            );
        }
        // Display Only One Prediction
        result.innerHTML = `
            <div>
                <div class="predicted-digit">
                    ${data.digit}
                </div>
                <div class="probability">
                    ${data.probability}%
                </div>
                <p class="prediction-label">
                    Predicted Digit
                </p>
            </div>
        `;
    } catch (error) {
        result.innerHTML = `
            <p class="error-message">
                ${error.message}
            </p>
        `;
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "Predict";
    }
});
// Initialize Canvas
clearCanvas();