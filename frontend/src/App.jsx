import { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState("");

  const API_URL = "http://127.0.0.1:8000";

  // ============================================================
  // LOAD HISTORY ON START
  // ============================================================

  useEffect(() => {
    fetchHistory();
  }, []);

  // ============================================================
  // FILE SELECTION
  // ============================================================

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
  };

  // ============================================================
  // PREDICTION
  // ============================================================

  const handlePredict = async () => {
    if (!selectedFile) {
      setError("Please select a plant leaf image first.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || "Prediction failed.");
      }

      setResult(data);

      // Refresh prediction history
      fetchHistory();
    } catch (err) {
      console.error("Prediction error:", err);

      setError(
        err.message ||
          "Unable to connect to the AgriGuard AI server."
      );
    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // FETCH HISTORY
  // ============================================================

  const fetchHistory = async () => {
    setHistoryLoading(true);

    try {
      const response = await fetch(`${API_URL}/history`);

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.message || "Unable to load prediction history."
        );
      }

      setHistory(data.history || []);
    } catch (err) {
      console.error("History error:", err);
    } finally {
      setHistoryLoading(false);
    }
  };

  // ============================================================
  // CONFIDENCE
  // ============================================================

  const confidenceValue =
    result?.confidence ??
    result?.prediction?.confidence ??
    0;

  const confidencePercentage = confidenceValue * 100;

  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div className="app">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="header">
        <div className="brand">

          <div className="logo">
            🌱
          </div>

          <div>
            <h1>AgriGuard AI</h1>

            <p>
              Intelligent Smart Agriculture Platform
            </p>
          </div>

        </div>

        <div className="status">
          <span className="status-dot"></span>
          AI System
        </div>
      </header>

      {/* ======================================================
          NAVIGATION
      ====================================================== */}

      <nav className="navigation">

        <a href="#detector">
          Disease Detection
        </a>

        <a href="#history">
          Prediction History
        </a>

      </nav>

      <main className="main-container">

        {/* ====================================================
            HERO
        ==================================================== */}

        <section
          className="hero"
          id="detector"
        >

          <span className="badge">
            AI-Powered Plant Health
          </span>

          <h2>
            Detect Plant Diseases
            <br />
            <span>with AI</span>
          </h2>

          <p>
            Upload a clear image of a plant leaf and
            AgriGuard AI will analyze it using our
            deep-learning model.
          </p>

        </section>

        {/* ====================================================
            UPLOAD SECTION
        ==================================================== */}

        <section className="upload-card">

          <div className="section-title">

            <span>01</span>

            <h3>
              Upload Plant Image
            </h3>

          </div>

          <label className="upload-area">

            {preview ? (

              <div className="preview-container">

                <img
                  src={preview}
                  alt="Selected plant leaf"
                  className="preview-image"
                />

                <div className="change-image">
                  Click to choose another image
                </div>

              </div>

            ) : (

              <div className="upload-content">

                <div className="upload-icon">
                  📷
                </div>

                <h3>
                  Upload a plant leaf image
                </h3>

                <p>
                  Click here to browse your image
                </p>

                <span>
                  JPG, JPEG, PNG or WEBP
                </span>

              </div>

            )}

            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
            />

          </label>

          {selectedFile && (

            <div className="file-name">
              📄 {selectedFile.name}
            </div>

          )}

          <button
            className="analyze-button"
            onClick={handlePredict}
            disabled={!selectedFile || loading}
          >

            {loading ? (

              <>
                <span className="spinner"></span>
                Analyzing Image...
              </>

            ) : (

              <>
                🔍 Analyze Plant
              </>

            )}

          </button>

          {error && (

            <div className="error-box">
              ⚠️ {error}
            </div>

          )}

        </section>

        {/* ====================================================
            RESULT SECTION
        ==================================================== */}

        {result && (

          <section className="result-section">

            <div className="section-title">

              <span>02</span>

              <h3>
                Analysis Result
              </h3>

            </div>

            {/* ==================================================
                INVALID INPUT
            ================================================== */}

            {result.status === "invalid_input" && (

              <div className="uncertain-card">

                <div className="uncertain-icon">
                  🚫
                </div>

                <div className="uncertain-content">

                  <h2>
                    Invalid Image
                  </h2>

                  <p>
                    This image does not appear to contain
                    a plant leaf.
                  </p>

                  {result.message && (

                    <p>
                      {result.message}
                    </p>

                  )}

                  {result.leaf_confidence !== null &&
                    result.leaf_confidence !== undefined && (

                    <div className="confidence-box">

                      <div className="confidence-header">

                        <span>
                          Leaf Detection Confidence
                        </span>

                        <strong>
                          {(
                            result.leaf_confidence * 100
                          ).toFixed(2)}
                          %
                        </strong>

                      </div>

                      <div className="progress-background">

                        <div
                          className="progress uncertain-progress"
                          style={{
                            width: `${Math.min(
                              result.leaf_confidence * 100,
                              100
                            )}%`,
                          }}
                        ></div>

                      </div>

                    </div>

                  )}

                  <div className="review-message">

                    <strong>
                      What to do:
                    </strong>

                    <ul>

                      <li>
                        Upload an image containing
                        a plant leaf.
                      </li>

                      <li>
                        Make sure the leaf is clearly
                        visible.
                      </li>

                      <li>
                        Avoid screenshots, documents,
                        objects or scenery.
                      </li>

                    </ul>

                  </div>

                </div>

              </div>

            )}

            {/* ==================================================
                UNCERTAIN
            ================================================== */}

            {result.status === "uncertain" && (

              <div className="uncertain-card">

                <div className="uncertain-icon">
                  ⚠️
                </div>

                <div className="uncertain-content">

                  <h2>
                    Unable to confidently identify
                    the disease
                  </h2>

                  <p>
                    The AI model is not sufficiently
                    confident about this image.
                  </p>

                  <div className="confidence-box">

                    <div className="confidence-header">

                      <span>
                        Model Confidence
                      </span>

                      <strong>
                        {confidencePercentage.toFixed(2)}%
                      </strong>

                    </div>

                    <div className="progress-background">

                      <div
                        className="progress uncertain-progress"
                        style={{
                          width: `${Math.min(
                            confidencePercentage,
                            100
                          )}%`,
                        }}
                      ></div>

                    </div>

                  </div>

                  <div className="review-message">

                    <strong>
                      What to do:
                    </strong>

                    <ul>

                      <li>
                        Upload a clearer image of
                        the affected leaf.
                      </li>

                      <li>
                        Make sure the affected area
                        is clearly visible.
                      </li>

                      <li>
                        Consult an agricultural expert
                        if uncertainty continues.
                      </li>

                    </ul>

                  </div>

                  {result.message && (

                    <p>
                      {result.message}
                    </p>

                  )}

                </div>

              </div>

            )}

            {/* ==================================================
                HEALTHY LEAF
            ================================================== */}

            {result.status === "healthy" && (

              <div className="prediction-card">

                <div className="prediction-header">

                  <div>

                    <span className="result-label">
                      HEALTHY LEAF
                    </span>

                    <h2>
                      No Disease Detected
                    </h2>

                    <p className="crop-name">
                      🌿 {result.crop || "Plant"}
                    </p>

                    <p className="scientific-name">
                      The uploaded leaf appears healthy
                    </p>

                  </div>

                  <div className="confidence-circle">

                    <strong>
                      {confidencePercentage.toFixed(1)}%
                    </strong>

                    <span>
                      Confidence
                    </span>

                  </div>

                </div>

                <div className="confidence-section">

                  <div className="confidence-header">

                    <span>
                      Model Confidence
                    </span>

                    <strong>
                      {confidencePercentage.toFixed(2)}%
                    </strong>

                  </div>

                  <div className="progress-background">

                    <div
                      className="progress"
                      style={{
                        width: `${Math.min(
                          confidencePercentage,
                          100
                        )}%`,
                      }}
                    ></div>

                  </div>

                  <small>
                    Disease confidence threshold: 75%
                  </small>

                </div>

                <div className="info-grid">

                  <div className="info-card">

                    <div className="info-icon">
                      🌿
                    </div>

                    <h3>
                      Result
                    </h3>

                    <p>
                      No disease was detected in the
                      uploaded {result.crop || "plant"} leaf image.
                    </p>

                  </div>

                  <div className="info-card">

                    <div className="info-icon">
                      🔍
                    </div>

                    <h3>
                      Leaf Validation
                    </h3>

                    <p>
                      The image was successfully identified
                      as containing a plant leaf.
                    </p>

                  </div>

                  <div className="info-card">

                    <div className="info-icon">
                      ℹ️
                    </div>

                    <h3>
                      Important
                    </h3>

                    <p>
                      A healthy classification does not
                      guarantee that the plant is completely
                      free from every possible condition.
                    </p>

                  </div>

                </div>

              </div>

            )}

            {/* ==================================================
                CONFIDENT DISEASE PREDICTION
            ================================================== */}

            {result.status === "prediction" && (

              <div className="prediction-card">

                <div className="prediction-header">

                  <div>

                    <span className="result-label">
                      DETECTED CONDITION
                    </span>

                    <h2>
                      {result.disease ||
                        result.prediction?.disease}
                    </h2>

                    <p className="crop-name">
                      🌾{" "}
                      {result.crop ||
                        result.prediction?.crop ||
                        "Crop information available"}
                    </p>

                    {(result.scientific_name ||
                      result.prediction?.scientific_name) && (

                      <p className="scientific-name">
                        {result.scientific_name ||
                          result.prediction?.scientific_name}
                      </p>

                    )}

                  </div>

                  <div className="confidence-circle">

                    <strong>
                      {confidencePercentage.toFixed(1)}%
                    </strong>

                    <span>
                      Confidence
                    </span>

                  </div>

                </div>

                {/* CONFIDENCE */}

                <div className="confidence-section">

                  <div className="confidence-header">

                    <span>
                      Model Confidence
                    </span>

                    <strong>
                      {confidencePercentage.toFixed(2)}%
                    </strong>

                  </div>

                  <div className="progress-background">

                    <div
                      className="progress"
                      style={{
                        width: `${Math.min(
                          confidencePercentage,
                          100
                        )}%`,
                      }}
                    ></div>

                  </div>

                  <small>
                    Confidence threshold: 75%
                  </small>

                </div>

                {/* DISEASE INFORMATION */}

                {result.disease_information && (

                  <div className="info-grid">

                    <div className="info-card">

                      <div className="info-icon">
                        🔬
                      </div>

                      <h3>
                        Description
                      </h3>

                      <p>
                        {result.disease_information.description}
                      </p>

                    </div>

                    <div className="info-card">

                      <div className="info-icon">
                        🩺
                      </div>

                      <h3>
                        Symptoms
                      </h3>

                      <p>
                        {result.disease_information.symptoms}
                      </p>

                    </div>

                    <div className="info-card">

                      <div className="info-icon">
                        🛡️
                      </div>

                      <h3>
                        Prevention
                      </h3>

                      <p>
                        {result.disease_information.prevention}
                      </p>

                    </div>

                  </div>

                )}

                {/* MANAGEMENT GUIDELINES */}

                {result.management_guidelines &&
                  result.management_guidelines.length > 0 && (

                  <div className="management-section">

                    <div className="management-heading">

                      <div className="management-icon">
                        ✓
                      </div>

                      <div>

                        <h3>
                          Management Guidelines
                        </h3>

                        <p>
                          Based on verified agricultural
                          sources
                        </p>

                      </div>

                    </div>

                    {result.management_guidelines.map(
                      (guideline, index) => (

                      <div
                        className="guideline"
                        key={index}
                      >

                        <p>
                          {guideline.guideline}
                        </p>

                        <div className="source">

                          <strong>
                            Source:
                          </strong>

                          <span>
                            {guideline.source}
                          </span>

                        </div>

                        <div className="verified">

                          ✓ Last verified:{" "}
                          {guideline.last_verified}

                        </div>

                      </div>

                    ))}

                  </div>

                )}

                {/* DISCLAIMER */}

                {result.disclaimer && (

                  <div className="disclaimer">

                    <strong>
                      Important:
                    </strong>{" "}

                    {result.disclaimer}

                  </div>

                )}

              </div>

            )}

          </section>

        )}

        {/* ====================================================
            HISTORY
        ==================================================== */}

        <section
          className="history-section"
          id="history"
        >

          <div className="section-title">

            <span>03</span>

            <h3>
              Prediction History
            </h3>

          </div>

          <div className="history-card">

            <div className="history-header">

              <div>

                <h2>
                  Previous Analyses
                </h2>

                <p>
                  Your recent plant health analyses
                </p>

              </div>

              <button
                className="refresh-button"
                onClick={fetchHistory}
                disabled={historyLoading}
              >
                {historyLoading
                  ? "Loading..."
                  : "↻ Refresh"}
              </button>

            </div>

            {history.length === 0 ? (

              <div className="empty-history">

                <div>
                  📋
                </div>

                <h3>
                  No prediction history yet
                </h3>

                <p>
                  Your analyzed plant images will
                  appear here.
                </p>

              </div>

            ) : (

              <div className="history-list">

                {history.map((item) => {

                  const isPrediction =
                    item.status === "prediction";

                  const isHealthy =
                    item.status === "healthy";

                  const isInvalid =
                    item.status === "invalid_input";

                  const isUncertain =
                    item.status === "uncertain";

                  return (

                    <div
                      className="history-item"
                      key={item.id}
                    >

                      <div className="history-main">

                        <div className="history-icon">

                          {isPrediction
                            ? "🌿"
                            : isHealthy
                            ? "🌱"
                            : isInvalid
                            ? "🚫"
                            : "⚠️"}

                        </div>

                        <div>

                          <h3>

                            {isPrediction
                              ? item.disease
                              : isHealthy
                              ? "Healthy Leaf"
                              : isInvalid
                              ? "Invalid Image"
                              : "Needs Review"}

                          </h3>

                          <p>

                            {item.crop
                              ? item.crop
                              : isHealthy
                              ? "No disease detected"
                              : isInvalid
                              ? "Non-leaf image rejected"
                              : "Disease not confidently identified"}

                          </p>

                          <span>
                            {item.filename}
                          </span>

                        </div>

                      </div>

                      <div className="history-meta">

                        <div
                          className={
                            isPrediction || isHealthy
                              ? "history-status prediction-status"
                              : "history-status uncertain-status"
                          }
                        >

                          {isPrediction
                            ? "Detected"
                            : isHealthy
                            ? "Healthy"
                            : isInvalid
                            ? "Rejected"
                            : "Uncertain"}

                        </div>

                        <strong>

                          {isInvalid
                            ? "—"
                            : item.confidence !== null &&
                              item.confidence !== undefined
                            ? `${(
                                Number(item.confidence) * 100
                              ).toFixed(2)}%`
                            : "—"}

                        </strong>

                        <small>

                          {item.predicted_at
                            ? new Date(
                                item.predicted_at
                              ).toLocaleString()
                            : ""}

                        </small>

                      </div>

                    </div>

                  );
                })}

              </div>

            )}

          </div>

        </section>

      </main>

      {/* ======================================================
          FOOTER
      ====================================================== */}

      <footer className="footer">

        <p>
          AgriGuard AI • Intelligent Smart Agriculture
        </p>

        <p>
          AI-assisted decision support for plant health
        </p>

      </footer>

    </div>
  );
}

export default App;