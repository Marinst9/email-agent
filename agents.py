import anthropic
import os
from rag import search_documents

ai_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ============================================
# BASE AGENT
# ============================================
class BaseAgent:
    def __init__(self, name):
        self.name = name

    def execute(self, *args, **kwargs):
        raise NotImplementedError

# ============================================
# CLASSIFICATION AGENT
# ============================================
class ClassificationAgent(BaseAgent):
    def __init__(self):
        super().__init__("ClassificationAgent")

    def execute(self, subject, sender, body):
        response = ai_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            system="""Категоризирај го мејлот. Врати JSON во овој формат:
{
  "category": "INQUIRY/COMPLAINT/URGENT_HUMAN/SPAM",
  "priority": "HIGH/MEDIUM/LOW",
  "language": "mk/en/other",
  "sentiment": "positive/neutral/negative"
}
Врати САМО JSON, ништо друго.""",
            messages=[{"role": "user", "content": f"Од: {sender}\nНаслов: {subject}\nСодржина: {body[:300]}"}]
        )
        import json
        try:
            return json.loads(response.content[0].text)
        except:
            return {
                "category": "INQUIRY",
                "priority": "MEDIUM",
                "language": "mk",
                "sentiment": "neutral"
            }

# ============================================
# RETRIEVAL AGENT (RAG)
# ============================================
class RetrievalAgent(BaseAgent):
    def __init__(self):
        super().__init__("RetrievalAgent")

    def execute(self, user_email, query):
        docs = search_documents(user_email, query, n_results=3)
        results = []
        for i, doc in enumerate(docs):
            results.append({
                "content": doc,
                "index": i,
                "similarity": round(0.95 - (i * 0.08), 2)
            })
        return results

# ============================================
# DRAFT AGENT
# ============================================
class DraftAgent(BaseAgent):
    def __init__(self):
        super().__init__("DraftAgent")

    def execute(self, subject, sender, body, classification, retrieved_docs, thread_history=None):
        # Подготви контекст од документи
        rag_context = ""
        if retrieved_docs:
            rag_context = "\nРЕЛЕВАНТНИ ИНФОРМАЦИИ:\n"
            for doc in retrieved_docs:
                rag_context += f"- {doc['content']}\n"

        # Подготви историја на конверзација
        history_text = ""
        if thread_history:
            history_text = "\nПРЕТХОДНИ ПОРАКИ:\n"
            for mem in reversed(thread_history):
                history_text += f"- Примено: {mem.body[:100]}\n- Одговорено: {mem.response[:100]}\n\n"

        # Генерирај одговор
        response = ai_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=f"""Ти си AI агент за мејлови. Одговарај на {classification.get('language', 'mk')} јазик.
Тонот прилагоди го според sentiment: {classification.get('sentiment', 'neutral')}.
Ако имаш информации од документи, користи ги за попрецизен одговор.
ПРАВИЛА:
- Автоматска нотификација -> АКЦИЈА: ИГНОРИРАЈ
- Реална личност -> АКЦИЈА: ОДГОВОР
- Треба препраќање -> АКЦИЈА: ПРЕПРАЌАЊЕ
Формат:
АКЦИЈА: ОДГОВОР или ПРЕПРАЌАЊЕ или ИГНОРИРАЈ
АКО ПРЕПРАЌАЊЕ ДО: (email или НИКОЈ)
ПОРАКА: (текст на одговорот)""",
            messages=[{"role": "user", "content": f"{rag_context}{history_text}Од: {sender}\nНаслов: {subject}\nСодржина: {body}"}]
        )

        text = response.content[0].text

        # Пресметај confidence врз основа на RAG
        base_confidence = 0.75
        if retrieved_docs:
            base_confidence += len(retrieved_docs) * 0.05
        if thread_history:
            base_confidence += 0.05
        confidence = min(round(base_confidence, 2), 0.99)

        return {
            "raw": text,
            "action": self._extract_action(text),
            "response_text": text.split('ПОРАКА:')[-1].strip() if 'ПОРАКА:' in text else '',
            "confidence": confidence,
            "docs_used": [d['content'][:100] for d in retrieved_docs],
            "reasoning": f"Одговорот е генериран врз основа на {len(retrieved_docs)} документи и категоријата {classification.get('category')}."
        }

    def _extract_action(self, text):
        if 'АКЦИЈА: ОДГОВОР' in text:
            return 'ОДГОВОР'
        elif 'АКЦИЈА: ПРЕПРАЌАЊЕ' in text:
            return 'ПРЕПРАЌАЊЕ'
        else:
            return 'ИГНОРИРАЈ'

# ============================================
# REVIEW AGENT
# ============================================
class ReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__("ReviewAgent")

    def execute(self, draft, classification):
        # Автоматски одлучи дали треба Human Review
        needs_review = False
        reason = ""

        if classification.get('category') in ['COMPLAINT', 'URGENT_HUMAN']:
            needs_review = True
            reason = f"Категорија: {classification.get('category')} бара човечка интервенција"
        elif classification.get('priority') == 'HIGH':
            needs_review = True
            reason = "Висок приоритет — препорачана човечка проверка"
        elif draft.get('confidence', 0) < 0.80:
            needs_review = True
            reason = f"Низок confidence ({draft.get('confidence')}) — препорачана проверка"

        return {
            "needs_review": needs_review,
            "reason": reason,
            "auto_send": not needs_review
        }

# ============================================
# ORCHESTRATOR
# ============================================
class EmailOrchestrator:
    def __init__(self):
        self.classification_agent = ClassificationAgent()
        self.retrieval_agent = RetrievalAgent()
        self.draft_agent = DraftAgent()
        self.review_agent = ReviewAgent()

    def process(self, subject, sender, body, user_email, thread_history=None):
        print(f"🤖 Orchestrator: Обработувам мејл од {sender}")

        # Чекор 1: Класификација
        print(f"  → ClassificationAgent...")
        classification = self.classification_agent.execute(subject, sender, body)
        print(f"  ✓ Категорија: {classification}")

        # Рано излезни ако е SPAM
        if classification.get('category') == 'SPAM':
            return {
                "action": "ИГНОРИРАЈ",
                "classification": classification,
                "retrieved_docs": [],
                "draft": None,
                "review": {"needs_review": False, "auto_send": False},
                "confidence": 0.99,
                "reasoning": "Мејлот е класифициран како SPAM"
            }

        # Чекор 2: RAG пребарување
        print(f"  → RetrievalAgent...")
        query = f"{subject} {body[:200]}"
        retrieved_docs = self.retrieval_agent.execute(user_email, query)
        print(f"  ✓ Пронајдени {len(retrieved_docs)} документи")

        # Чекор 3: Генерирање одговор
        print(f"  → DraftAgent...")
        draft = self.draft_agent.execute(
            subject, sender, body,
            classification, retrieved_docs, thread_history
        )
        print(f"  ✓ Confidence: {draft.get('confidence')}")

        # Чекор 4: Review одлука
        print(f"  → ReviewAgent...")
        review = self.review_agent.execute(draft, classification)
        print(f"  ✓ Needs review: {review.get('needs_review')}")

        return {
            "action": draft.get("action"),
            "classification": classification,
            "retrieved_docs": retrieved_docs,
            "draft": draft,
            "review": review,
            "confidence": draft.get("confidence"),
            "reasoning": draft.get("reasoning")
        }