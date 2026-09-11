from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)


# =========================================================
# SAMPLE DATA FOR ML MODEL
# =========================================================

np.random.seed(42)

data = []

for i in range(1000):

    previous_marks = np.random.randint(40, 96)
    test_marks = np.random.randint(35, 96)
    assignment_marks = np.random.randint(40, 101)

    study_hours = np.random.uniform(1, 6)
    difficulty = np.random.randint(1, 6)
    revision = np.random.randint(1, 6)
    exam_days = np.random.randint(3, 31)

    performance = (
        previous_marks * 0.30
        + test_marks * 0.25
        + assignment_marks * 0.15
        + study_hours * 3
        + revision * 2
        - difficulty * 1.5
        + (30 - exam_days) * 0.2
        + np.random.normal(0, 3)
    )

    performance = max(0, min(100, performance))

    data.append([
        previous_marks,
        test_marks,
        assignment_marks,
        study_hours,
        difficulty,
        revision,
        exam_days,
        performance
    ])


columns = [
    "previous_marks",
    "test_marks",
    "assignment_marks",
    "study_hours",
    "difficulty",
    "revision",
    "exam_days",
    "performance"
]

df = pd.DataFrame(data, columns=columns)


# =========================================================
# TRAIN MODEL
# =========================================================

X = df.drop("performance", axis=1)
y = df["performance"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# =========================================================
# FORMAT MINUTES
# =========================================================

def format_minutes(minutes):

    minutes = int(minutes)

    if minutes < 60:
        return f"{minutes} min"

    hours = minutes // 60
    remaining = minutes % 60

    if remaining == 0:
        return f"{hours} hr"

    return f"{hours} hr {remaining} min"


# =========================================================
# SUBJECT PRIORITY
# =========================================================

def calculate_priority(
        marks,
        difficulty,
        revision):

    marks_factor = 100 - marks

    difficulty_factor = difficulty * 20

    revision_factor = (6 - revision) * 20

    priority = (
        marks_factor * 0.50
        + difficulty_factor * 0.25
        + revision_factor * 0.25
    )

    return round(
        max(0, min(100, priority))
    )


# =========================================================
# WEAK AREA DETECTION
# =========================================================

def detect_weak_area(
        marks,
        test,
        assignment,
        difficulty,
        revision):

    scores = {

        "Previous Performance":
            marks,

        "Test Performance":
            test,

        "Assignment Performance":
            assignment,

        "Concept Understanding":
            100 - ((difficulty - 1) * 15),

        "Revision":
            revision * 20

    }

    weakest = min(
        scores,
        key=scores.get
    )

    return weakest


# =========================================================
# STUDY STRATEGY
# =========================================================

def get_strategy(
        marks,
        difficulty,
        revision):

    if marks < 50:

        return (
            "Focus on basic concepts and "
            "practice simple questions first."
        )

    if difficulty >= 4:

        return (
            "Use concept learning followed "
            "by problem-solving practice."
        )

    if revision <= 2:

        return (
            "Spend more time revising formulas, "
            "definitions and important concepts."
        )

    if marks >= 80:

        return (
            "Use short revision sessions and "
            "solve previous question papers."
        )

    return (
        "Balance concept learning, practice "
        "and revision."
    )


# =========================================================
# GENERATE STUDY PLAN
# =========================================================

def generate_study_plan(
        subjects,
        daily_minutes):

    total_priority = sum(
        subject["priority"]
        for subject in subjects
    )

    if total_priority == 0:
        total_priority = 1


    for subject in subjects:

        allocated = (
            subject["priority"]
            / total_priority
        ) * daily_minutes

        allocated = max(
            10,
            round(allocated / 5) * 5
        )

        subject["minutes"] = allocated


    # Reduce if total exceeds available time

    while sum(
        s["minutes"]
        for s in subjects
    ) > daily_minutes:

        largest = max(
            subjects,
            key=lambda x: x["minutes"]
        )

        if largest["minutes"] <= 10:
            break

        largest["minutes"] -= 5


    # Distribute remaining time

    while (
        sum(s["minutes"] for s in subjects)
        + 5
        <= daily_minutes
    ):

        highest = max(
            subjects,
            key=lambda x: x["priority"]
        )

        highest["minutes"] += 5


    for subject in subjects:

        subject["time"] = format_minutes(
            subject["minutes"]
        )


# =========================================================
# DAY-WISE PLAN
# =========================================================

def create_day_plan(
        subjects,
        exam_days):

    days = []

    number_of_days = min(
        exam_days,
        7
    )

    for day in range(
        1,
        number_of_days + 1
    ):

        day_subjects = []

        sorted_subjects = sorted(
            subjects,
            key=lambda x: x["priority"],
            reverse=True
        )

        for subject in sorted_subjects:

            day_subjects.append({

                "name": subject["name"],

                "time": subject["time"]

            })


        days.append({

            "day": day,

            "subjects": day_subjects

        })


    return days


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    student_name = request.form.get(
        "student_name"
    )

    available_hours = float(
        request.form.get(
            "available_hours"
        )
    )

    exam_days = int(
        request.form.get(
            "exam_days"
        )
    )

    daily_minutes = int(
        available_hours * 60
    )


    subject_names = request.form.getlist(
        "subject_name[]"
    )

    marks_list = request.form.getlist(
        "marks[]"
    )

    test_list = request.form.getlist(
        "test[]"
    )

    assignment_list = request.form.getlist(
        "assignment[]"
    )

    difficulty_list = request.form.getlist(
        "difficulty[]"
    )

    revision_list = request.form.getlist(
        "revision[]"
    )


    subjects = []


    # =====================================================
    # PROCESS SUBJECTS
    # =====================================================

    for i in range(
        len(subject_names)
    ):

        if not subject_names[i].strip():
            continue


        marks = float(
            marks_list[i]
        )

        test = float(
            test_list[i]
        )

        assignment = float(
            assignment_list[i]
        )

        difficulty = int(
            difficulty_list[i]
        )

        revision = int(
            revision_list[i]
        )


        priority = calculate_priority(
            marks,
            difficulty,
            revision
        )


        weak_area = detect_weak_area(
            marks,
            test,
            assignment,
            difficulty,
            revision
        )


        strategy = get_strategy(
            marks,
            difficulty,
            revision
        )


        subjects.append({

            "name": subject_names[i],

            "marks": marks,

            "test": test,

            "assignment": assignment,

            "difficulty": difficulty,

            "revision": revision,

            "priority": priority,

            "weak_area": weak_area,

            "strategy": strategy

        })


    if len(subjects) == 0:

        return "Please add at least one subject."


    # =====================================================
    # AVERAGES
    # =====================================================

    average_marks = np.mean([
        s["marks"]
        for s in subjects
    ])

    average_test = np.mean([
        s["test"]
        for s in subjects
    ])

    average_assignment = np.mean([
        s["assignment"]
        for s in subjects
    ])

    average_difficulty = np.mean([
        s["difficulty"]
        for s in subjects
    ])

    average_revision = np.mean([
        s["revision"]
        for s in subjects
    ])


    # =====================================================
    # ML PREDICTION
    # =====================================================

    input_data = pd.DataFrame([{

        "previous_marks":
            average_marks,

        "test_marks":
            average_test,

        "assignment_marks":
            average_assignment,

        "study_hours":
            available_hours,

        "difficulty":
            average_difficulty,

        "revision":
            average_revision,

        "exam_days":
            exam_days

    }])


    predicted_performance = model.predict(
        input_data
    )[0]


    predicted_performance = round(
        max(
            0,
            min(
                100,
                predicted_performance
            )
        ),
        1
    )


    # =====================================================
    # PERFORMANCE CATEGORY
    # =====================================================

    if predicted_performance >= 85:

        category = "Excellent"

    elif predicted_performance >= 70:

        category = "Good"

    elif predicted_performance >= 55:

        category = "Average"

    else:

        category = "Needs Improvement"


    # =====================================================
    # SORT BY PRIORITY
    # =====================================================

    subjects.sort(
        key=lambda x: x["priority"],
        reverse=True
    )


    # =====================================================
    # GENERATE PLAN
    # =====================================================

    generate_study_plan(
        subjects,
        daily_minutes
    )


    # =====================================================
    # DAY-WISE PLAN
    # =====================================================

    day_plan = create_day_plan(
        subjects,
        exam_days
    )


    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    recommendations = []


    highest = subjects[0]


    recommendations.append(
        f"Give highest priority to "
        f"{highest['name']}."
    )


    for subject in subjects:

        if subject["marks"] < 60:

            recommendations.append(
                f"Improve {subject['name']} "
                f"through additional practice."
            )


        if subject["difficulty"] >= 4:

            recommendations.append(
                f"Spend extra concept-learning "
                f"time on {subject['name']}."
            )


    if exam_days <= 7:

        recommendations.append(
            "Exam is close. Focus on revision "
            "and previous question papers."
        )

    elif exam_days <= 15:

        recommendations.append(
            "Balance concept learning with revision."
        )

    else:

        recommendations.append(
            "Use this period to strengthen "
            "weak concepts."
        )


    # =====================================================
    # TOTAL TIME
    # =====================================================

    total_minutes = sum(
        s["minutes"]
        for s in subjects
    )


    total_time = format_minutes(
        total_minutes
    )


    # =====================================================
    # ADAPTIVE REPLANNING MESSAGE
    # =====================================================

    adaptive_message = (
        "After each study session, update your "
        "completed topics and the planner can "
        "increase time for unfinished subjects."
    )


    # =====================================================
    # RETURN RESULT
    # =====================================================

    return render_template(

        "result.html",

        student_name=student_name,

        predicted_performance=
            predicted_performance,

        category=category,

        subjects=subjects,

        recommendations=
            recommendations,

        daily_minutes=
            daily_minutes,

        total_time=
            total_time,

        exam_days=
            exam_days,

        day_plan=
            day_plan,

        adaptive_message=
            adaptive_message
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )