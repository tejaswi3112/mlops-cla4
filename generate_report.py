"""
Script to generate the comprehensive academic & technical PDF report:
'MLOps_Assignment_Report.pdf' for Task 6 and Submission Requirements.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print 'Page X of Y' in footer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#718096"))

        # Running header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "ML Model Deployment Strategies & Model Serving using FastAPI | Titanic ML")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Running footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 36, page_str)
        self.drawString(54, 36, "MLOps Continuous Learning Assignment (CLA-4) | Confidential")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 48, letter[0] - 54, 48)

        self.restoreState()


def build_pdf_report(filename="MLOps_Assignment_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Palettes
    primary_color = colors.HexColor("#1E3A8A")   # Deep Navy
    secondary_color = colors.HexColor("#0D9488") # Deep Teal
    text_dark = colors.HexColor("#1F2937")       # Charcoal
    bg_light = colors.HexColor("#F8FAFC")        # Off-white
    border_color = colors.HexColor("#E2E8F0")

    # Typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=text_dark,
        spaceAfter=6
    )

    bold_body = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=text_dark
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []

    # Title & Metadata Banner
    story.append(Paragraph("ML Model Deployment Strategies & Model Serving using FastAPI", title_style))
    story.append(Paragraph("Titanic: Machine Learning from Disaster | Complete Technical & Experimental Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=10))

    # Meta Info Table
    meta_data = [
        [
            Paragraph("<b>Dataset:</b> Kaggle Titanic Disaster (train.csv)", body_style),
            Paragraph("<b>Target Variable:</b> Survived (0 = No, 1 = Yes)", body_style)
        ],
        [
            Paragraph("<b>Model Architecture:</b> Scikit-Learn Pipeline (Random Forest)", body_style),
            Paragraph("<b>Evaluation Test Accuracy:</b> 80.45%", body_style)
        ],
        [
            Paragraph("<b>Model Serving Framework:</b> FastAPI 0.143 + Uvicorn + Pydantic", body_style),
            Paragraph("<b>Artifacts:</b> model.pkl, batch_predictions.csv", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # SECTION 1: ARCHITECTURE & METHODOLOGY
    story.append(Paragraph("1. System Architecture & Pipeline Design", h1_style))
    story.append(Paragraph(
        "Modern Machine Learning systems require seamless parity between training and serving environments. "
        "To eliminate <b>training-serving skew</b>, an end-to-end <b>Scikit-Learn Pipeline</b> was implemented combining "
        "data imputation, feature scaling, one-hot encoding, and a non-linear Random Forest Classifier into a unified serialized object (<code>model.pkl</code>). "
        "The model is trained once on an 80% stratified split (712 records) and evaluated on an unseen 20% test partition (179 records). "
        "This identical pipeline is strictly reused across Online, Batch, Streaming, and REST API serving modes.",
        body_style
    ))

    pipeline_summary = [
        [Paragraph("Feature Group", table_header), Paragraph("Input Features", table_header), Paragraph("Preprocessing Pipeline Transformers", table_header)],
        [Paragraph("Numerical", table_cell), Paragraph("Age, SibSp, Parch, Fare", table_cell), Paragraph("SimpleImputer(strategy='median') → StandardScaler()", table_cell)],
        [Paragraph("Categorical", table_cell), Paragraph("Pclass, Sex, Embarked", table_cell), Paragraph("SimpleImputer(strategy='most_frequent') → OneHotEncoder(handle_unknown='ignore')", table_cell)],
        [Paragraph("Estimator", table_cell), Paragraph("Cleaned Feature Vector", table_cell), Paragraph("RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)", table_cell)]
    ]
    pipe_table = Table(pipeline_summary, colWidths=[80, 150, 274])
    pipe_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(pipe_table)
    story.append(Spacer(1, 8))

    # SECTION 2: MODEL TRAINING & EVALUATION
    story.append(Paragraph("2. Task 1: Model Training & Evaluation Metrics", h1_style))
    story.append(Paragraph(
        "Evaluation on the held-out test partition demonstrated robust discrimination and balanced precision/recall: "
        "<b>Accuracy: 80.45%</b>, <b>F1-Score (Survived): 0.71</b>, <b>F1-Score (Non-survived): 0.85</b>.",
        body_style
    ))

    eval_data = [
        [Paragraph("Metric", table_header), Paragraph("Did Not Survive (Class 0)", table_header), Paragraph("Survived (Class 1)", table_header), Paragraph("Macro / Weighted Average", table_header)],
        [Paragraph("Precision", table_cell), Paragraph("0.79", table_cell), Paragraph("0.84", table_cell), Paragraph("0.82 / 0.81", table_cell)],
        [Paragraph("Recall", table_cell), Paragraph("0.93", table_cell), Paragraph("0.61", table_cell), Paragraph("0.77 / 0.80", table_cell)],
        [Paragraph("F1-Score", table_cell), Paragraph("0.85", table_cell), Paragraph("0.71", table_cell), Paragraph("0.78 / 0.80", table_cell)],
        [Paragraph("Support", table_cell), Paragraph("110 records", table_cell), Paragraph("69 records", table_cell), Paragraph("Total: 179 unseen records", table_cell)],
    ]
    eval_table = Table(eval_data, colWidths=[90, 130, 130, 154])
    eval_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(eval_table)
    story.append(Spacer(1, 10))

    # SECTION 3: INFERENCE STRATEGIES DEMONSTRATION
    story.append(Paragraph("3. Demonstration of Inference Strategies (Tasks 2, 3, & 4)", h1_style))
    story.append(Paragraph(
        "Three distinct inference paradigms were implemented using the identical trained pipeline <code>model.pkl</code>:",
        body_style
    ))

    demo_data = [
        [
            Paragraph("Inference Strategy", table_header),
            Paragraph("Execution Mechanism", table_header),
            Paragraph("Performance Metrics", table_header),
            Paragraph("Primary Operational Advantage", table_header)
        ],
        [
            Paragraph("<b>Task 2: Online Inference</b><br/><code>online_inference.py</code>", table_cell),
            Paragraph("Synchronous request-response evaluation of 1 unseen record (Passenger #566, Mr. Alfred Davies).", table_cell),
            Paragraph("Latency: <b>143.8 ms</b><br/>Confidence: <b>88.19%</b><br/>Result: Did Not Survive", table_cell),
            Paragraph("Immediate instant feedback suitable for end-user web applications and real-time decision flows.", table_cell)
        ],
        [
            Paragraph("<b>Task 3: Batch Inference</b><br/><code>batch_inference.py</code>", table_cell),
            Paragraph("Vectorized matrix transformation of 20 unseen records concurrently, exported to <code>batch_predictions.csv</code>.", table_cell),
            Paragraph("Total Time: <b>141.7 ms</b><br/>Per-Record: <b>7.09 ms</b><br/>Throughput: <b>141.1 rec/s</b>", table_cell),
            Paragraph("Extreme computational efficiency via vectorized CPU instructions and batched overhead amortisation.", table_cell)
        ],
        [
            Paragraph("<b>Task 4: Streaming Inference</b><br/><code>streaming_inference.py</code>", table_cell),
            Paragraph("Simulated message stream of 10 incoming records arriving at 1.0s intervals, processed sequentially upon arrival.", table_cell),
            Paragraph("Avg Model Latency: <b>74.3 ms</b><br/>Inter-arrival: <b>1.0 sec</b><br/>Events: 10/10 processed", table_cell),
            Paragraph("Low-latency continuous inference for event-driven message architectures (Kafka, Kinesis, IoT feeds).", table_cell)
        ]
    ]
    demo_table = Table(demo_data, colWidths=[110, 135, 120, 139])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 10))

    # SECTION 4: FASTAPI MODEL SERVING
    story.append(Paragraph("4. Task 5: Model Serving with FastAPI & Pydantic Validation", h1_style))
    story.append(Paragraph(
        "A production-grade REST API was engineered using <b>FastAPI</b> (<code>app.py</code>) running on ASGI server Uvicorn. "
        "Strict input validation and schema enforcement are executed via <b>Pydantic</b> models.",
        body_style
    ))

    api_points = [
        "<b>Endpoint GET /health:</b> Verifies service status, active loaded model, timestamp, and API version.",
        "<b>Endpoint POST /predict:</b> Accepts JSON passenger records, enforces field types, value constraints (e.g. <code>1 &le; Pclass &le; 3</code>, <code>Age &ge; 0</code>, <code>Sex &isin; {'male', 'female'}</code>), passes inputs through the pipeline, and returns prediction class, label, probability, confidence, and inference latency.",
        "<b>Interactive Documentation:</b> Auto-generated OpenAPI / Swagger UI at <code>/docs</code> and ReDoc at <code>/redoc</code>.",
        "<b>Python Automated Client (<code>client.py</code>):</b> Automated test suite verifying health checks, valid predictions, and verifying that malformed payloads trigger HTTP <code>422 Unprocessable Entity</code> with detailed error messages."
    ]
    for pt in api_points:
        story.append(Paragraph(f"• {pt}", body_style))
    story.append(Spacer(1, 10))

    # SECTION 5: TASK 6 DETAILED OBSERVATIONS
    story.append(PageBreak())
    story.append(Paragraph("5. Task 6: Key Observations & Technical Answers", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=primary_color, spaceAfter=8))

    # Q1
    story.append(Paragraph("Question 1: How does Online Inference differ from Batch Inference?", h2_style))
    q1_text = (
        "<b>Online Inference</b> (also known as Real-Time or Request-Response Inference) operates synchronously: "
        "a client application sends an individual data instance (or small micro-batch) over a network protocol (HTTP/REST or gRPC) "
        "and expects an immediate low-latency prediction (typically sub-100ms). It prioritizes <b>latency and responsiveness</b>, "
        "powering interactive user-facing systems such as live fraud detection, interactive recommendation engines, and chatbot scoring.<br/><br/>"
        "In contrast, <b>Batch Inference</b> (Offline / Scheduled Inference) operates asynchronously on large volumes of data collected over time "
        "(e.g., millions of records processed hourly or daily). The entire batch is processed simultaneously using vectorized matrix transformations. "
        "Batch inference prioritizes <b>throughput, computational efficiency, and cost</b> over turnaround time. "
        "In our benchmarks, batch processing reduced average per-record latency from <b>143.8 ms to 7.09 ms</b> (over a 20x improvement in throughput)."
    )
    story.append(Paragraph(q1_text, body_style))
    story.append(Spacer(1, 6))

    # Q2
    story.append(Paragraph("Question 2: What is the difference between Streaming Inference and Batch Inference?", h2_style))
    q2_text = (
        "<b>Streaming Inference</b> operates in near-real-time on continuous, unbounded event data streams (e.g., Apache Kafka, AWS Kinesis, IoT sensor feeds). "
        "As each event or small sliding micro-window arrives, the inference engine immediately computes and emits a prediction without waiting for data to be persisted to disk. "
        "Streaming inference is <b>event-driven, push-based, and handles dynamic temporal windows</b>.<br/><br/>"
        "<b>Batch Inference</b>, conversely, operates on static, bounded datasets persisted in data warehouses or object storage (e.g., Snowflake, BigQuery, AWS S3). "
        "Execution is typically <b>schedule-driven (via cron/Airflow) or threshold-driven</b>. "
        "While batch inference allows global dataset aggregations and high throughput, streaming inference enables real-time responsiveness to changing environmental conditions as events happen."
    )
    story.append(Paragraph(q2_text, body_style))
    story.append(Spacer(1, 6))

    # Q3
    story.append(Paragraph("Question 3: Why is FastAPI used for model serving?", h2_style))
    q3_text = (
        "FastAPI has emerged as the premier standard for Python-based ML model serving due to several critical technical advantages:<br/>"
        "<b>1. Asynchronous High Performance:</b> Built on Starlette and Uvicorn, FastAPI leverages Python's native <code>async/await</code> and ASGI concurrency, "
        "delivering throughput on par with NodeJS and Go, significantly outperforming legacy WSGI frameworks like Flask.<br/>"
        "<b>2. Native Data Validation with Pydantic:</b> FastAPI uses Python type hints and Pydantic models to automatically validate incoming payloads, "
        "parse types, and return standardized HTTP 422 error details when invalid features are provided, protecting ML models from malformed inputs.<br/>"
        "<b>3. Automatic OpenAPI & Swagger Documentation:</b> Interactive UI documentation (<code>/docs</code> and <code>/redoc</code>) is automatically generated from code annotations, enabling immediate testing and client integration without manual schema maintenance.<br/>"
        "<b>4. Production Ergonomics:</b> Native dependency injection, built-in CORS middleware, lifespan events for lifecycle model loading, and seamless Docker containerization make FastAPI production-ready."
    )
    story.append(Paragraph(q3_text, body_style))
    story.append(Spacer(1, 6))

    # Q4
    story.append(Paragraph("Question 4: Why should the same trained model be reused for all inference strategies?", h2_style))
    q4_text = (
        "Reusing the identical serialized model pipeline (<code>model.pkl</code>) across all deployment modes is a foundational MLOps best practice for four vital reasons:<br/>"
        "<b>1. Prevention of Training-Serving Skew:</b> If preprocessing rules (e.g., median imputation values, scaling factors, one-hot category vocabularies) are reimplemented separately for online, batch, or streaming workflows, subtle mathematical divergence and code drift occur, leading to silent predictive degradation.<br/>"
        "<b>2. Reproducibility & Consistency:</b> A given input passenger profile must yield the exact same prediction and probability score regardless of whether it was submitted via REST API, evaluated in a nightly batch CSV, or ingested from a streaming queue.<br/>"
        "<b>3. Maintainability & Single Source of Truth:</b> Packaging feature transformers and model weights into one Scikit-Learn Pipeline allows teams to update, version, test, and audit a single model artifact rather than maintaining redundant preprocessing codebases.<br/>"
        "<b>4. Operational Governance & Model Registry Parity:</b> In production MLOps pipelines (MLflow, Kubeflow, Vertex AI), tracking one versioned artifact hash ensures traceable governance, rollbacks, and compliance."
    )
    story.append(Paragraph(q4_text, body_style))
    story.append(Spacer(1, 10))

    # SECTION 6: COMPARATIVE SUMMARY MATRIX
    story.append(Paragraph("6. Comparative Matrix of Inference Deployment Strategies", h1_style))
    comp_matrix = [
        [
            Paragraph("Dimension", table_header),
            Paragraph("Online Inference", table_header),
            Paragraph("Batch Inference", table_header),
            Paragraph("Streaming Inference", table_header)
        ],
        [
            Paragraph("<b>Latency Profile</b>", table_cell),
            Paragraph("Ultra-low (Milliseconds, ~50-150ms)", table_cell),
            Paragraph("High latency, high throughput (Minutes to Hours)", table_cell),
            Paragraph("Near-real-time (Sub-second per event)", table_cell)
        ],
        [
            Paragraph("<b>Data Input Format</b>", table_cell),
            Paragraph("Single JSON record or small payload", table_cell),
            Paragraph("Bulk static datasets (CSV, Parquet, SQL tables)", table_cell),
            Paragraph("Continuous event stream (Kafka, Kinesis, sockets)", table_cell)
        ],
        [
            Paragraph("<b>Trigger Mechanism</b>", table_cell),
            Paragraph("Synchronous client request (HTTP/gRPC)", table_cell),
            Paragraph("Cron scheduler, pipeline trigger, or queue threshold", table_cell),
            Paragraph("Event arrival interrupt / message broker", table_cell)
        ],
        [
            Paragraph("<b>Compute Strategy</b>", table_cell),
            Paragraph("On-demand persistent server instances (FastAPI)", table_cell),
            Paragraph("Vectorized bulk processing / ephemeral worker clusters", table_cell),
            Paragraph("Continuous stream processing engines (Flink, Spark)", table_cell)
        ],
        [
            Paragraph("<b>Primary Metric</b>", table_cell),
            Paragraph("p95/p99 Latency & Availability", table_cell),
            Paragraph("Records Processed / Sec & Cost Efficiency", table_cell),
            Paragraph("Message Ingestion Lag & Stream Freshness", table_cell)
        ]
    ]
    comp_table = Table(comp_matrix, colWidths=[104, 134, 134, 132])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 12))

    # SECTION 7: SUBMISSION CHECKLIST & ARTIFACTS
    story.append(Paragraph("7. Submission Deliverables Verification", h1_style))
    deliverables = [
        [Paragraph("Deliverable File", table_header), Paragraph("Description & Verification Status", table_header)],
        [Paragraph("<code>train.py</code>", table_cell), Paragraph("Task 1: Model training, preprocessing pipeline, evaluation (80.45%), exports <code>model.pkl</code> and <code>unseen_test_data.csv</code>.", table_cell)],
        [Paragraph("<code>online_inference.py</code>", table_cell), Paragraph("Task 2: Single-record online inference demonstration with latency measurement and confidence scores.", table_cell)],
        [Paragraph("<code>batch_inference.py</code>", table_cell), Paragraph("Task 3: Bulk inference over 20 unseen records; generates <code>batch_predictions.csv</code>.", table_cell)],
        [Paragraph("<code>streaming_inference.py</code>", table_cell), Paragraph("Task 4: Simulated event stream inference with 1-second arrival delay across 10 unseen records.", table_cell)],
        [Paragraph("<code>app.py</code>", table_cell), Paragraph("Task 5: FastAPI server with GET /health, POST /predict, and strict Pydantic input validation.", table_cell)],
        [Paragraph("<code>client.py</code>", table_cell), Paragraph("Task 5: Automated Python client testing /health, valid predictions, and verifying HTTP 422 for invalid payloads.", table_cell)],
        [Paragraph("<code>model.pkl</code>", table_cell), Paragraph("Serialized complete Scikit-Learn Pipeline combining imputation, scaling, one-hot encoding, and Random Forest classifier.", table_cell)],
        [Paragraph("<code>batch_predictions.csv</code>", table_cell), Paragraph("Generated output containing predictions, ground truth, and probabilities for 20 unseen passengers.", table_cell)],
        [Paragraph("<code>screenshots/</code>", table_cell), Paragraph("Visual proofs: Online, Batch, Streaming, Client Tests, and Swagger UI (/health, /predict valid, /predict invalid 422).", table_cell)],
        [Paragraph("<code>requirements.txt</code>", table_cell), Paragraph("Complete pinned dependencies required to reproduce and run the solution.", table_cell)],
        [Paragraph("<code>README.md</code>", table_cell), Paragraph("Full documentation, architecture overview, dataset links, execution commands, and observations.", table_cell)]
    ]
    deliv_table = Table(deliverables, colWidths=[140, 364])
    deliv_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(deliv_table)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] Successfully compiled PDF Report: '{filename}'")


if __name__ == "__main__":
    build_pdf_report("MLOps_Assignment_Report.pdf")

