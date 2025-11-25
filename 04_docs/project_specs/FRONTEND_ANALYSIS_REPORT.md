# 프론트엔드 코드 분석 리포트

- **작성 일시:** 2025-11-24 05:00 (KST)
- **분석 대상:** `/home/pencilfoxs/00_new/3_Docent_frontend_1117`
- **목적:** RAG 시스템 백엔드 연결을 위한 프론트엔드 구조 파악

---

## 📋 1. 기술 스택 (Tech Stack)

| 항목 | 기술 | 버전 |
|:---|:---|:---|
| **프레임워크** | Next.js | 15.3.3 |
| **언어** | TypeScript | 5.x |
| **UI 라이브러리** | React | 18.3.1 |
| **스타일링** | Tailwind CSS | 3.4.1 |
| **UI 컴포넌트** | Radix UI | 최신 |
| **AI 통합** | Genkit (Google) | 1.20.0 |
| **개발 포트** | 9002 | (dev 스크립트) |

---

## 🏗️ 2. 프로젝트 구조

```
3_Docent_frontend_1117/
├── src/
│   ├── app/
│   │   ├── page.tsx              # 메인 페이지 (ChatView 렌더링)
│   │   ├── layout.tsx            # 루트 레이아웃
│   │   └── actions.ts            # ⭐ Server Actions (백엔드 통신)
│   ├── components/
│   │   ├── chat/
│   │   │   ├── chat-view.tsx     # 채팅 뷰 (메인)
│   │   │   ├── chat-list.tsx     # 채팅 리스트
│   │   │   ├── chat-message.tsx  # 개별 메시지
│   │   │   └── chat-input-form.tsx # 입력 폼
│   │   └── ui/                   # shadcn/ui 컴포넌트들
│   ├── ai/
│   │   ├── genkit.ts             # Genkit 설정
│   │   ├── dev.ts                # Genkit 개발 서버
│   │   └── flows/
│   │       ├── royal-answer-based-on-location.ts  # ⭐ 핵심 Flow (미구현)
│   │       ├── generate-visual-explanations.ts
│   │       ├── receive-smart-nearby-recommendations.ts
│   │       ├── speech-to-text.ts
│   │       └── text-to-speech.ts
│   └── lib/
│       ├── types.ts              # 타입 정의
│       ├── constants.ts          # 상수 (Persona 등)
│       └── utils.ts
└── package.json
```

---

## 🔌 3. 현재 백엔드 통신 구조

### 3.1 Server Actions 패턴 (Next.js)

프론트엔드는 **Next.js Server Actions**를 사용하여 백엔드와 통신합니다.

**핵심 파일:** `src/app/actions.ts`

```typescript
// 사용자 질문 → 답변 생성
export async function getRoyalAnswer(formData: FormData) {
  const validatedData = royalAnswerSchema.parse(Object.fromEntries(formData));
  const result = await royalAnswerBasedOnLocation(validatedData);
  return { success: true, answer: result.answer };
}
```

### 3.2 API 요청 형식 분석

#### 입력 (Request)
```typescript
const royalAnswerSchema = z.object({
  question: z.string(),                    // 사용자 질문
  location: z.string(),                    // 위치 정보
  historicalFigurePersona: z.string(),     // 페르소나 (세종대왕, 이순신 등)
  photoDataUri: z.string().optional(),     // 사진 데이터 (선택)
  language: z.string(),                    // 언어 (ko, en, ja, zh-CN)
});
```

#### 출력 (Response)
```typescript
{
  success: true,
  answer: string  // 생성된 답변
}
// 또는
{
  success: false,
  error: string   // 에러 메시지
}
```

### 3.3 현재 구현 상태

**문제점 발견:**
- ✅ `actions.ts`: Server Action 함수는 정의되어 있음
- ❌ `royal-answer-based-on-location.ts`: **파일이 비어있음 (미구현)**
- ❓ `chat-view.tsx`, `chat-input-form.tsx`: 파일이 비어있거나 미완성 상태

**현재 상황:**
- 프론트엔드 구조는 완성되어 있지만, **실제 AI Flow 로직이 구현되지 않은 상태**
- Genkit을 사용하려는 의도가 있지만, 실제 RAG 시스템과 연결되지 않음

---

## 🎯 4. RAG 시스템 연결 방안

### 4.1 현재 구조 vs 필요한 구조

#### 현재 (Genkit 기반, 미구현)
```
프론트엔드 (Next.js)
  ↓ Server Action
  ↓ Genkit Flow
  ↓ Google AI (Gemini API)
```

#### 목표 (RAG 시스템 연결)
```
프론트엔드 (Next.js)
  ↓ Server Action (getRoyalAnswer)
  ↓ FastAPI 서버 (새로 구축)
  ↓ HistoryDocent 클래스 (RAG 시스템)
  ↓ BM25 검색 + LLM 생성
```

### 4.2 연결 전략 (2가지 옵션)

#### 옵션 1: Server Actions 유지 + FastAPI 백엔드 추가 (권장) ⭐

**장점:**
- 프론트엔드 코드 수정 최소화
- Next.js의 Server Actions 패턴 유지
- 백엔드와 프론트엔드 분리 (확장성 좋음)

**구조:**
```typescript
// src/app/actions.ts (수정)
export async function getRoyalAnswer(formData: FormData) {
  const validatedData = royalAnswerSchema.parse(Object.fromEntries(formData));
  
  // FastAPI 백엔드 호출
  const response = await fetch('http://localhost:8000/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: validatedData.question,
      // ... 기타 정보
    }),
  });
  
  const result = await response.json();
  return { success: true, answer: result.answer };
}
```

**필요한 작업:**
1. ✅ FastAPI 서버 구축 (`/api/query` 엔드포인트)
2. ✅ `HistoryDocent` 통합 클래스 작성
3. ✅ Server Action에서 FastAPI 호출하도록 수정

---

#### 옵션 2: Genkit Flow를 RAG로 대체

**장점:**
- 현재 Genkit 구조 활용
- Google Cloud와 통합 용이

**단점:**
- Genkit에서 Python RAG 시스템 직접 호출 어려움
- 결국 API 서버가 필요 (옵션 1과 동일)

---

### 4.3 최종 추천 방안: 옵션 1 (FastAPI 백엔드)

**이유:**
1. **프론트엔드 수정 최소화**: Server Action만 수정하면 됨
2. **백엔드 독립성**: RAG 시스템을 별도 서버로 운영 가능
3. **확장성**: 나중에 다른 프론트엔드 연결 가능
4. **디버깅 용이**: 백엔드와 프론트엔드 분리로 문제 파악 쉬움

---

## 📝 5. 필요한 작업 체크리스트

### 5.1 백엔드 작업 (우선순위 높음)

- [ ] **FastAPI 서버 작성**
  - 엔드포인트: `POST /api/query`
  - 요청 형식: `{ question: string, ... }`
  - 응답 형식: `{ answer: string, sources?: [...] }`

- [ ] **HistoryDocent 통합 클래스 작성**
  - `query(question: str) -> str` 메서드
  - 검색 + 생성 통합
  - 멀티턴 지원 (선택)

- [ ] **CORS 설정**
  - Next.js 프론트엔드(9002) → FastAPI(8000) 허용

### 5.2 프론트엔드 작업 (최소 수정)

- [ ] **Server Action 수정** (`src/app/actions.ts`)
  - `getRoyalAnswer` 함수에서 FastAPI 호출
  - 에러 처리 강화

- [ ] **Chat 컴포넌트 확인**
  - `chat-view.tsx`, `chat-input-form.tsx` 구현 상태 확인
  - 필요시 완성

---

## 🔍 6. 발견된 주요 특징

### 6.1 페르소나 시스템
프론트엔드는 **역사 인물 페르소나**를 지원합니다:
- 세종대왕, 이순신, 신사임당, 황진이, 장금이, 초랭이, 역사 가이드

**연결 시 고려사항:**
- RAG 시스템에서 페르소나 정보를 받아서 프롬프트에 반영할 수 있음
- 예: "세종대왕 페르소나로 답변해주세요"

### 6.2 다국어 지원
- 한국어, 영어, 일본어, 중국어 지원
- RAG 시스템 응답도 다국어로 제공 가능 (현재는 한국어만)

### 6.3 추가 기능
- 음성 입력/출력 (STT/TTS)
- 이미지 업로드 (선택)
- 위치 기반 답변

**현재 RAG 시스템과의 연관성:**
- 위치, 이미지는 무시하고 **질문(question)**만 사용 가능
- TTS는 답변 생성 후 프론트엔드에서 처리

---

## 💡 7. 연결 시 주의사항

### 7.1 API 스펙 일치
- 프론트엔드가 기대하는 형식: `{ success: boolean, answer?: string, error?: string }`
- FastAPI 응답을 이 형식에 맞춰야 함

### 7.2 비동기 처리
- LLM 답변 생성은 시간이 걸림 (5-10초)
- 프론트엔드에 로딩 상태 표시 필요 (이미 구현되어 있을 가능성)

### 7.3 에러 처리
- 네트워크 오류, 타임아웃 등 처리
- 사용자에게 친화적인 에러 메시지

---

## ✅ 8. 결론 및 다음 단계

### 현재 상태 요약
- ✅ 프론트엔드 UI 구조는 완성되어 있음
- ❌ 실제 RAG 시스템과 연결되어 있지 않음
- ✅ 연결을 위한 구조는 이미 준비되어 있음 (Server Actions)

### 다음 단계
1. **FastAPI 백엔드 서버 구축** (우선)
2. **HistoryDocent 통합 클래스 작성**
3. **Server Action 수정** (FastAPI 호출)
4. **연결 테스트 및 디버깅**

### 예상 작업 시간
- FastAPI 서버 + 통합 클래스: **2-3시간**
- 프론트엔드 수정: **30분-1시간**
- 테스트 및 디버깅: **1-2시간**
- **총 예상 시간: 4-6시간**

---

**작성자:** AI Assistant  
**최종 업데이트:** 2025-11-24 05:00 (KST)

