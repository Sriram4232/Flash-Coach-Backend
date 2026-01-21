import json
from typing import List, Dict, Any

PROBLEM_KEYWORDS = {
    'classroom_management': [
        'disruptive', 'behavior', 'attention', 'talking', 'discipline', 
        'distracted', 'noise', 'chaos', 'control', 'restless', 'fighting',
        'misbehaving', 'not listening', 'off-task', 'rowdy'
    ],
    'conceptual_misunderstanding': [
        'understand', 'concept', 'confused', 'explain', 'grasp', 
        'comprehend', 'difficulty', 'struggling', 'mistake', 'wrong answer',
        'don\'t get it', 'learning', 'math', 'reading', 'science'
    ],
    'multi_level_activity': [
        'different levels', 'mixed abilities', 'advanced students', 'struggling students',
        'differentiation', 'varied', 'diverse', 'some students', 'others',
        'fast learners', 'slow learners', 'inclusion'
    ],
    'assessment_anxiety': [
        'test', 'exam', 'anxiety', 'nervous', 'stressed', 'worried',
        'performance', 'pressure', 'fear', 'panic', 'blank', 'freeze',
        'assessment', 'quiz'
    ]
}

def classify_problem(query: str) -> str:
    """Classify the problem type based on keywords in the query."""
    lower_query = query.lower()
    max_score = 0
    problem_type = 'other'
    
    for type_name, keywords in PROBLEM_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower_query)
        if score > max_score:
            max_score = score
            problem_type = type_name
    
    return problem_type

def select_strategies(strategies: List[Dict], problem_type: str, 
                     past_interactions: List[Dict], confidence_level: str = 'medium'):
    """Select appropriate strategies based on context and history."""
    # Filter strategies by problem type
    matching_strategies = [s for s in strategies if s.get('type') == problem_type]
    
    if not matching_strategies:
        matching_strategies = strategies
    
    # Get failed strategy IDs
    failed_strategy_ids = set()
    strategy_failures = {}
    
    for interaction in past_interactions:
        if interaction.get('feedback') == 'failed' and interaction.get('advice'):
            advice_list = interaction['advice']
            for advice in advice_list:
                strategy_id = advice.get('strategy_id')
                if strategy_id:
                    failed_strategy_ids.add(strategy_id)
                    strategy_failures[strategy_id] = strategy_failures.get(strategy_id, 0) + 1
    
    # Filter out strategies that failed twice
    available_strategies = [
        s for s in matching_strategies 
        if strategy_failures.get(s['_id'], 0) < 2
    ]
    
    # If all strategies have failed twice, reset but mark for escalation
    should_escalate = False
    if not available_strategies:
        available_strategies = matching_strategies
        should_escalate = True
    
    # Sort by difficulty based on confidence level
    difficulty_order = []
    if confidence_level == 'low':
        difficulty_order = ['simple', 'moderate', 'advanced']
    elif confidence_level == 'high':
        difficulty_order = ['advanced', 'moderate', 'simple']
    else:
        difficulty_order = ['moderate', 'simple', 'advanced']
    
    def difficulty_score(strategy):
        difficulty = strategy.get('difficulty', 'moderate')
        return difficulty_order.index(difficulty) if difficulty in difficulty_order else len(difficulty_order)
    
    available_strategies.sort(key=difficulty_score)
    
    return {
        'strategies': available_strategies[:3],
        'should_escalate': should_escalate
    }

def build_prompt(query: str, problem_type: str, selected_strategies: List[Dict], 
                teacher: Dict, classroom: Dict, past_interactions: List[Dict]) -> str:
    """Build the prompt for the LLM."""
    recent_feedback = []
    for interaction in past_interactions[-3:]:
        if interaction.get('feedback') and interaction.get('feedback') != 'pending':
            recent_feedback.append(f"- Problem: {interaction.get('problem_type', 'unknown')}, "
                                 f"Feedback: {interaction.get('feedback')}")
    
    strategy_context = "\n".join([
        f"- {s['title']}: {s['description']}" 
        for s in selected_strategies
    ]) if selected_strategies else 'Use your expertise to suggest appropriate strategies'
    
    # Build teacher context
    teacher_name = teacher.get('name', 'Teacher') if teacher else 'Teacher'
    teacher_grade = teacher.get('grade', classroom.get('grade_level', 'Not specified') if classroom else 'Not specified') if teacher else 'Not specified'
    teacher_confidence = teacher.get('confidence_level', 'medium') if teacher else 'medium'
    teacher_experience = teacher.get('years_experience', 'Not specified') if teacher else 'Not specified'
    
    # Build classroom context
    student_count = classroom.get('student_count', 'Not specified') if classroom else 'Not specified'
    multi_level = 'Yes' if classroom and classroom.get('multi_level_flag') else 'No'
    
    prompt_text = (
        f"TEACHER CONTEXT:\n"
        f"- Name: {teacher_name}\n"
        f"- Grade Level: {teacher_grade}\n"
        f"- Confidence Level: {teacher_confidence}\n"
        f"- Experience: {teacher_experience} years\n\n"

        f"CLASSROOM CONTEXT:\n"
        f"- Students: {student_count}\n"
        f"- Multi-level class: {multi_level}\n\n"

        f"PROBLEM TYPE IDENTIFIED:\n"
        f"{problem_type.replace('_', ' ').upper()}\n\n"

        f"RECENT FEEDBACK ON PAST ADVICE:\n"
        f"{chr(10).join(recent_feedback) if recent_feedback else 'No recent feedback'}\n\n"

        f"REFERENCE STRATEGIES (OPTIONAL CONTEXT - DO NOT COPY VERBATIM):\n"
        f"{strategy_context}\n\n"

        f"TEACHER'S QUESTION:\n"
        f"\"{query}\""
    )

    return prompt_text

def check_escalation_needed(interactions: List[Dict]) -> bool:
    """Check if escalation is needed based on recent failures."""
    recent_interactions = interactions[-10:]
    failed_count = sum(1 for i in recent_interactions if i.get('feedback') == 'failed')
    return failed_count >= 3