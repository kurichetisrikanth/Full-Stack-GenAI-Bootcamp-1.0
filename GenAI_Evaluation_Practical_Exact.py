# -*- coding: utf-8 -*-

ORIGINAL_PRACTICAL = r'''Bilkul bhai. **Ab exact practical karte hain** — ek chhota **Employee Policy RAG application** banayenge aur usi application ko 4 angles se evaluate karenge:

```text id="2558w1"
Application
   ↓
LLM Evaluation
   ↓
RAG Evaluation
   ↓
Agent Evaluation
   ↓
Production Evaluation
```

Main pehle **metrics manually implement** kar raha hoon. Framework baad me lagayenge. Ye teaching ke liye zyada useful hai, kyunki student ko samajh aayega score actually calculate kaise hota hai.

Current LangChain Python API me `ChatOpenAI`, `InMemoryVectorStore`, `create_agent` aur `@tool` available hain, so neeche ka structure current LangChain style follow karta hai. ([docs.langchain.com](https://docs.langchain.com/oss/python/integrations/chat/openai?utm_source=chatgpt.com))

---

# Practical Project: GenAI Evaluation System

## Step 1 — Installation

```bash id="rdwpo0"
pip install -U langchain langchain-openai pydantic pandas
```

Environment variable:

```bash id="qvqmps"
OPENAI_API_KEY=your_key
```

---

# Step 2 — Imports

```python id="zi7mxt"
import os
import math
import time
import pandas as pd

from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
```

Model initialize karo:

```python id="xwytks"
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model=os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small"
    )
)
```

---

# Step 3 — Mini Knowledge Base

Suppose company ki HR policy ko RAG me use karna hai.

```python id="suo6v4"
documents = [

    Document(
        page_content="""
        Employees are allowed to work from home
        for a maximum of 2 days per week.
        """,
        metadata={"doc_id": "remote_policy"}
    ),

    Document(
        page_content="""
        Every full-time employee receives
        24 paid leaves per calendar year.
        """,
        metadata={"doc_id": "leave_policy"}
    ),

    Document(
        page_content="""
        Employees can claim up to ₹3000 per month
        for internet reimbursement.
        """,
        metadata={"doc_id": "internet_policy"}
    ),

    Document(
        page_content="""
        The standard probation period
        for new employees is 6 months.
        """,
        metadata={"doc_id": "probation_policy"}
    ),

    Document(
        page_content="""
        Employees receive ₹1000 per month
        as mobile reimbursement.
        """,
        metadata={"doc_id": "mobile_policy"}
    )
]
```

---

# Step 4 — Vector Store

Demo ke liye `InMemoryVectorStore` use karenge. LangChain docs ke according ye in-process ephemeral vector store hai aur `similarity_search()` support karta hai. ([docs.langchain.com](https://docs.langchain.com/oss/python/integrations/vectorstores?utm_source=chatgpt.com))

```python id="b6vx7v"
vector_store = InMemoryVectorStore(
    embedding=embeddings
)

ids = [
    doc.metadata["doc_id"]
    for doc in documents
]

vector_store.add_documents(
    documents=documents,
    ids=ids
)
```

---

# Step 5 — Retriever

```python id="vkch6q"
def retrieve(query: str, k: int = 3):

    docs = vector_store.similarity_search(
        query,
        k=k
    )

    return docs
```

Test:

```python id="vqgwnr"
results = retrieve(
    "How many paid leaves do employees receive?"
)

for doc in results:
    print(doc.metadata["doc_id"])
    print(doc.page_content)
    print()
```

Output ideally:

```text id="yo9dxn"
leave_policy
Every full-time employee receives
24 paid leaves per calendar year.

remote_policy
...

probation_policy
...
```

Ab yahin se evaluation shuru hoti hai.

---

# PART 1 — Retrieval Evaluation

Suppose user query:

```text id="h4hbrx"
How many paid leaves do employees receive?
```

Ground truth:

```python id="u49yxy"
relevant_docs = ["leave_policy"]
```

Retriever output:

```python id="vusqyj"
retrieved_docs = [
    "leave_policy",
    "probation_policy",
    "remote_policy"
]
```

Ab metrics calculate karte hain.

---

# 1. Precision@K

```python id="pj63un"
def precision_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
):

    retrieved_k = retrieved[:k]

    relevant_count = sum(
        doc in relevant
        for doc in retrieved_k
    )

    return relevant_count / k
```

Run:

```python id="2lx6cr"
score = precision_at_k(
    retrieved_docs,
    relevant_docs,
    k=3
)

print(score)
```

Output:

```text id="052h84"
0.3333
```

Kyun?

```text id="td8lxm"
Top 3 retrieved = 3 documents

Relevant = 1

Precision@3
= 1 / 3
= 0.33
```

---

# 2. Recall@K

```python id="pj0c60"
def recall_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
):

    retrieved_k = retrieved[:k]

    found_relevant = sum(
        doc in retrieved_k
        for doc in relevant
    )

    return found_relevant / len(relevant)
```

Run:

```python id="lew5s2"
recall = recall_at_k(
    retrieved_docs,
    relevant_docs,
    3
)

print(recall)
```

Output:

```text id="bpz6tc"
1.0
```

Because relevant document tha:

```text id="f0qply"
leave_policy
```

aur wo retrieve ho gaya.

---

# 3. Hit Rate

Very simple:

```python id="akf13w"
def hit_rate(
    retrieved: List[str],
    relevant: List[str],
    k: int
):

    retrieved_k = retrieved[:k]

    hit = any(
        doc in relevant
        for doc in retrieved_k
    )

    return 1 if hit else 0
```

```python id="gnkrsk"
print(
    hit_rate(
        retrieved_docs,
        relevant_docs,
        3
    )
)
```

Output:

```text id="531nch"
1
```

Matlab:

```text id="vldaes"
Relevant result mila?
YES → 1
NO  → 0
```

---

# 4. MRR — Mean Reciprocal Rank

Pehla relevant document kis position par mila?

```python id="3u1low"
def reciprocal_rank(
    retrieved: List[str],
    relevant: List[str]
):

    for rank, doc in enumerate(
        retrieved,
        start=1
    ):

        if doc in relevant:
            return 1 / rank

    return 0
```

Case 1:

```python id="eaav8i"
retrieved = [
    "leave_policy",
    "remote_policy",
    "internet_policy"
]
```

Relevant rank:

```text id="9iwryd"
1
```

Therefore:

```text id="s83p7q"
RR = 1/1 = 1
```

Code:

```python id="ek3gp7"
print(
    reciprocal_rank(
        retrieved,
        ["leave_policy"]
    )
)
```

Output:

```text id="l6q6jc"
1.0
```

Agar:

```python id="h3ce8d"
retrieved = [
    "remote_policy",
    "internet_policy",
    "leave_policy"
]
```

Then:

```text id="r10g9c"
RR = 1/3
   = 0.333
```

---

# 5. NDCG

Ab ranking quality evaluate karni hai.

```python id="a0ah2k"
def ndcg_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
):

    retrieved_k = retrieved[:k]

    dcg = 0

    for i, doc in enumerate(
        retrieved_k,
        start=1
    ):

        relevance = (
            1 if doc in relevant
            else 0
        )

        dcg += relevance / math.log2(i + 1)

    ideal_relevant_count = min(
        len(relevant),
        k
    )

    idcg = sum(
        1 / math.log2(i + 1)
        for i in range(
            1,
            ideal_relevant_count + 1
        )
    )

    if idcg == 0:
        return 0

    return dcg / idcg
```

```python id="2lk86d"
score = ndcg_at_k(
    [
        "remote_policy",
        "leave_policy",
        "internet_policy"
    ],
    ["leave_policy"],
    3
)

print(score)
```

Yahan relevant document rank 2 par hai, so score rank 1 ke comparison me kam aayega.

---

# Important Observation

Ye difference bahut important hai:

```text id="ky551v"
Recall@3 = 1

But

NDCG < 1
```

Meaning:

> Retriever ne correct document retrieve toh kiya, but usko ideal position par rank nahi kiya.

Exactly isi wajah se sirf Recall check karna enough nahi hota.

Ragas ke current docs bhi context precision ko relevant contexts ko higher rank karne ke perspective se evaluate karte hain aur context recall ko missing relevant information ke perspective se. ([docs.ragas.io](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/?utm_source=chatgpt.com))

---

# Step 6 — Complete RAG Pipeline

Ab retrieved documents LLM ko denge.

```python id="4qovr6"
def rag_answer(query: str, k: int = 3):

    retrieved_docs = retrieve(
        query,
        k=k
    )

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    prompt = f"""
You are an HR policy assistant.

Answer the question using ONLY
the provided context.

If the answer is not available in
the context, say:

"I don't know based on the provided context."

CONTEXT:
{context}

QUESTION:
{query}
"""

    start_time = time.perf_counter()

    response = llm.invoke(prompt)

    latency = (
        time.perf_counter()
        - start_time
    )

    return {
        "answer": response.content,
        "retrieved_docs": retrieved_docs,
        "latency": latency,
        "metadata": response.response_metadata
    }
```

Run:

```python id="fhhjlz"
result = rag_answer(
    "How many paid leaves do employees receive?"
)

print(result["answer"])
```

Expected:

```text id="nd14f5"
Employees receive 24 paid leaves
per calendar year.
```

---

# PART 2 — LLM / Generation Evaluation

Ab question hai:

```text id="7trxuo"
Answer sahi hai ya nahi?
Relevant hai?
Hallucinate toh nahi kiya?
Context follow kiya?
```

Yahan **LLM-as-a-Judge** use karenge.

---

# Step 7 — Evaluation Schema

```python id="96v61j"
class EvaluationScore(BaseModel):

    correctness: int = Field(
        ge=1,
        le=5
    )

    relevance: int = Field(
        ge=1,
        le=5
    )

    faithfulness: int = Field(
        ge=1,
        le=5
    )

    instruction_following: int = Field(
        ge=1,
        le=5
    )

    explanation: str
```

Structured evaluator:

```python id="n2s3ow"
judge = llm.with_structured_output(
    EvaluationScore
)
```

---

# Step 8 — Judge Function

```python id="hokeze"
def evaluate_answer(
    question,
    answer,
    reference_answer,
    context
):

    judge_prompt = f"""
You are evaluating an AI system.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

RETRIEVED CONTEXT:
{context}

AI ANSWER:
{answer}

Score the answer from 1 to 5
for the following:

Correctness:
Does the answer match the
reference answer?

Relevance:
Does the answer directly answer
the question?

Faithfulness:
Are all factual claims supported
by the retrieved context?

Instruction Following:
Did the assistant follow the
instruction to only use context?
"""

    return judge.invoke(
        judge_prompt
    )
```

---

# Step 9 — Test Dataset

Real evaluation ke liye ek question nahi, dataset chahiye.

```python id="zikrle"
eval_dataset = [

    {
        "question":
            "How many paid leaves do employees receive?",

        "expected_answer":
            "24 paid leaves per calendar year.",

        "relevant_docs":
            ["leave_policy"]
    },

    {
        "question":
            "How many work from home days are allowed?",

        "expected_answer":
            "Maximum 2 days per week.",

        "relevant_docs":
            ["remote_policy"]
    },

    {
        "question":
            "What is the monthly internet reimbursement limit?",

        "expected_answer":
            "₹3000 per month.",

        "relevant_docs":
            ["internet_policy"]
    },

    {
        "question":
            "How long is the probation period?",

        "expected_answer":
            "6 months.",

        "relevant_docs":
            ["probation_policy"]
    }
]
```

---

# Step 10 — Full RAG Evaluation Loop

Ab ye important code hai.

```python id="w0py3s"
evaluation_results = []

for sample in eval_dataset:

    query = sample["question"]

    result = rag_answer(
        query,
        k=3
    )

    answer = result["answer"]

    retrieved_documents = (
        result["retrieved_docs"]
    )

    retrieved_ids = [
        doc.metadata["doc_id"]
        for doc in retrieved_documents
    ]

    context = "\n".join(
        doc.page_content
        for doc in retrieved_documents
    )

    precision = precision_at_k(
        retrieved_ids,
        sample["relevant_docs"],
        3
    )

    recall = recall_at_k(
        retrieved_ids,
        sample["relevant_docs"],
        3
    )

    hit = hit_rate(
        retrieved_ids,
        sample["relevant_docs"],
        3
    )

    rr = reciprocal_rank(
        retrieved_ids,
        sample["relevant_docs"]
    )

    ndcg = ndcg_at_k(
        retrieved_ids,
        sample["relevant_docs"],
        3
    )

    judge_result = evaluate_answer(
        question=query,
        answer=answer,
        reference_answer=sample[
            "expected_answer"
        ],
        context=context
    )

    evaluation_results.append({

        "question":
            query,

        "answer":
            answer,

        "precision@3":
            precision,

        "recall@3":
            recall,

        "hit_rate":
            hit,

        "reciprocal_rank":
            rr,

        "ndcg@3":
            ndcg,

        "correctness":
            judge_result.correctness,

        "relevance":
            judge_result.relevance,

        "faithfulness":
            judge_result.faithfulness,

        "instruction_following":
            judge_result.instruction_following,

        "latency":
            result["latency"]
    })
```

---

# Step 11 — Evaluation Report

```python id="38xxdm"
df = pd.DataFrame(
    evaluation_results
)

print(df)
```

Example:

```text id="g0po39"
Question       Precision Recall MRR  Correctness Faithfulness
-------------------------------------------------------------
Paid leaves      0.33     1.0   1.0      5            5

WFH days         0.33     1.0   1.0      5            5

Internet limit   0.33     1.0   0.5      5            5

Probation        0.33     1.0   1.0      5            5
```

Overall report:

```python id="skhti8"
print(
    df[
        [
            "precision@3",
            "recall@3",
            "reciprocal_rank",
            "ndcg@3",
            "correctness",
            "relevance",
            "faithfulness",
            "latency"
        ]
    ].mean()
)
```

Now tumhare paas actual evaluation dashboard ka base aa gaya.

---

# PART 3 — Hallucination / Faithfulness Practical

Isko deliberately break karte hain.

Context:

```python id="o1f59o"
context = """
Employees receive 24 paid leaves
per year.
"""
```

Wrong LLM response:

```python id="0si1g6"
answer = """
Employees receive 24 paid leaves
and 10 sick leaves every year.
"""
```

Problem:

```text id="diu03l"
24 paid leaves → supported ✅

10 sick leaves → context me nahi hai ❌
```

Therefore faithfulness roughly:

```text id="1ysq9a"
Supported claims = 1
Total claims = 2

Faithfulness ≈ 1/2
             = 0.5
```

Ragas ka current faithfulness definition bhi essentially generated response ke claims ko retrieved context ke against check karta hai; score 0–1 hota hai. ([docs.ragas.io](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/?utm_source=chatgpt.com))

Ye classroom me bahut acha demo hai.

---

# PART 4 — Agent Evaluation

Ab RAG ke upar agent banate hain.

Agent ke paas do tools honge:

```text id="esmarb"
policy_search
multiply
```

Current LangChain agent API `create_agent()` aur `@tool` based tools support karta hai. ([docs.langchain.com](https://docs.langchain.com/oss/python/langchain/agents))

```python id="v3dmd3"
from langchain.agents import create_agent
from langchain.tools import tool
```

## Policy Search Tool

```python id="1quxcz"
@tool
def policy_search(query: str) -> str:
    """
    Search company HR policies.
    """

    docs = retrieve(
        query,
        k=2
    )

    return "\n".join(
        doc.page_content
        for doc in docs
    )
```

## Calculator Tool

```python id="rr5ih3"
@tool
def multiply(
    a: float,
    b: float
) -> float:
    """
    Multiply two numbers.
    """

    return a * b
```

Agent:

```python id="0hwszo"
agent = create_agent(
    model=llm,
    tools=[
        policy_search,
        multiply
    ],
    system_prompt="""
You are an HR assistant.

Use policy_search whenever
company policy information is needed.

Use multiply for multiplication.
"""
)
```

---

# Run Agent

Question:

```python id="sw6qlo"
query = """
Find the annual paid leave
entitlement and calculate how many
paid leaves an employee receives
over 3 years.
"""
```

Run:

```python id="mvkuyq"
agent_result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": query
        }
    ]
})
```

Agent ideally:

```text id="nfuxky"
User
 ↓
policy_search
 ↓
24 leaves/year
 ↓
multiply(24, 3)
 ↓
72
 ↓
Final Answer
```

---

# Extract Tool Calls

```python id="u4gclc"
tool_calls = []

for message in agent_result["messages"]:

    if hasattr(
        message,
        "tool_calls"
    ):

        if message.tool_calls:

            for call in message.tool_calls:

                tool_calls.append(
                    call["name"]
                )
```

```python id="owiy8z"
print(tool_calls)
```

Expected:

```text id="w9vqzl"
[
    "policy_search",
    "multiply"
]
```

---

# Tool Selection Accuracy

Ground truth:

```python id="lj4s0b"
expected_tools = [
    "policy_search",
    "multiply"
]
```

Actual:

```python id="pb2rt0"
actual_tools = tool_calls
```

Metric:

```python id="8y5gnc"
def tool_selection_accuracy(
    expected,
    actual
):

    correct = sum(
        tool in actual
        for tool in expected
    )

    return correct / len(expected)
```

```python id="ugc4ee"
print(
    tool_selection_accuracy(
        expected_tools,
        actual_tools
    )
)
```

Output:

```text id="hc5ozh"
1.0
```

Perfect.

---

# Agent Trajectory Evaluation

But suppose agent did:

```text id="aejufx"
policy_search
 ↓
policy_search
 ↓
policy_search
 ↓
multiply
 ↓
multiply
```

Answer correct ho sakta hai.

But agent inefficient hai.

Therefore:

```python id="pd70qv"
number_of_steps = len(
    tool_calls
)
```

Expected:

```text id="3gmgvs"
2
```

Actual maybe:

```text id="xkxg7r"
5
```

Then efficiency:

```python id="57ja2h"
def step_efficiency(
    expected_steps,
    actual_steps
):

    if actual_steps <= expected_steps:
        return 1.0

    return expected_steps / actual_steps
```

Example:

```python id="gfqqno"
print(
    step_efficiency(
        2,
        5
    )
)
```

Output:

```text id="87rkth"
0.4
```

LangSmith's current agent evaluation guidance similarly separates evaluation into **final response, trajectory, and individual agent steps/tool calls**. ([docs.langchain.com](https://docs.langchain.com/langsmith/evaluate-complex-agent?utm_source=chatgpt.com))

Ye Agent Evaluation ka core concept hai.

---

# PART 5 — Production Evaluation

Ab suppose production me 1000 requests aaye.

Har request par capture karo:

```python id="7phnzt"
production_metrics = {
    "latency_seconds":
        result["latency"],

    "input_tokens":
        result["metadata"]
        .get("token_usage", {})
        .get("prompt_tokens"),

    "output_tokens":
        result["metadata"]
        .get("token_usage", {})
        .get("completion_tokens"),

    "success":
        True,

    "user_feedback":
        None
}
```

Aggregate:

```python id="jzu063"
latencies = [
    1.2,
    1.4,
    2.1,
    1.8,
    5.4
]

average_latency = (
    sum(latencies)
    / len(latencies)
)

print(average_latency)
```

But production me sirf average nahi.

Check:

```text id="tqujcc"
P50 latency
P95 latency
P99 latency
Error rate
Token consumption
Cost/request
User thumbs up/down
Faithfulness trend
Task success rate
```

LangSmith currently separates **offline evaluation** for benchmarking/regression testing from **online evaluation** for production monitoring and anomaly detection. ([docs.langchain.com](https://docs.langchain.com/langsmith/evaluation-types?utm_source=chatgpt.com))

---

# Final Architecture

Ab tum jo system bana chuke ho usko dekho:

```text id="6h39ig"
                  USER QUERY
                      │
                      ▼
                 RETRIEVER
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
 Retrieval Evaluation        Retrieved Context
                              │
 Precision@K                  ▼
 Recall@K                    LLM
 Hit Rate                     │
 MRR                          ▼
 NDCG                    Generated Answer
                              │
                              ▼
                     Generation Evaluation
                              │
                         Correctness
                         Relevance
                         Faithfulness
                         Instruction Following


AGENT
  │
  ├── Tool Selection
  ├── Tool Call Accuracy
  ├── Trajectory
  ├── Task Success
  └── Step Efficiency


PRODUCTION
  │
  ├── Latency
  ├── Tokens
  ├── Cost
  ├── Error Rate
  ├── Safety
  └── User Feedback
```

## Aur ab framework kaha aata hai?

Ab jo humne manually banaya:

```text id="1y1m0l"
eval_dataset
      +
application
      +
metrics
      +
scores
```

**exactly isi concept ko Ragas, DeepEval aur LangSmith automate/manage karte hain.**

For example Ragas me current evaluation dataset `SingleTurnSample` objects contain `user_input`, `retrieved_contexts`, `response`, and optionally `reference`; Ragas built-in metrics me faithfulness, context precision, context recall, answer relevancy aur agent-oriented metrics available hain. ([docs.ragas.io](https://docs.ragas.io/en/stable/concepts/components/eval_dataset/?utm_source=chatgpt.com))

So main tumhare students ko sequence **aise hi padhata**:

```text id="5scq1m"
Practical 1
Manual LLM Evaluation
       ↓
Practical 2
Manual Retrieval Metrics
       ↓
Practical 3
RAG End-to-End Evaluation
       ↓
Practical 4
LLM-as-a-Judge
       ↓
Practical 5
Agent Evaluation
       ↓
Practical 6
Ragas
       ↓
Practical 7
DeepEval
       ↓
Practical 8
LangSmith Offline + Online Evaluation
```

Iske baad **RAGAS ka full practical karna sabse logical next step hai** — same dataset ko use karke `Faithfulness + Answer Relevancy + Context Precision + Context Recall` nikalenge, so manual calculation aur framework output side-by-side compare kar paoge.'''

if __name__ == "__main__":
    print(ORIGINAL_PRACTICAL)
