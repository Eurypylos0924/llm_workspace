# external_adapter/vector_db.py

import os
import re
from typing import Optional
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS



class VectorDBAdapter:
    """
    문서(PDF 등)를 텍스트 청크로 분할하여 FAISS Vector Store에 임베딩/저장하고
    유사도 검색(Similarity Search)을 수행하는 외부 어댑터 클래스입니다.
    """

    def __init__(
            self,
            folder_path: str = "./faiss_index",
            index_name: str = "financial_guide",
            nvidia_api_key: Optional[str] = None,
        ):
            self.folder_path = folder_path
            self.index_name = index_name
            self.api_key = nvidia_api_key or os.getenv("nvidiaapi_key")

            if not self.api_key:
                raise ValueError("nvidiaapi_key가 설정되어 있지 않습니다.")

            self.embeddings = NVIDIAEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b"),
                api_key=self.api_key
            )
            self.vector_store: Optional[FAISS] = None

    def _clean_text(self, text: str) -> str:
        """깨진 비표준 유니코드 기호 및 깨진 글자 정제 헬퍼 함수"""
        # 출력 불가능한 제어 문자 및 깨진 비표준 유니코드 범주 제거
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # 유효하지 않은 특수 유니코드 깨짐 기호 정제
        cleaned = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,?!()%\-\'\"/:]', ' ', cleaned)
        # 연속된 공백 하나로 축소
        return re.sub(r'\s+', ' ', cleaned).strip()

    def load_pdf_to_vector_db(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            return f"에러: 파일 경로를 찾을 수 없습니다: {pdf_path}"

        try:
            # 💡 1. PyMuPDFLoader로 데이터 로드 (한글 및 특수문자 지원 우수)
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()

            # 💡 2. 텍스트 정제 실행 (깨진 유니코드 제거)
            for doc in documents:
                doc.page_content = self._clean_text(doc.page_content)

            # 3. Document Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", ".", " ", ""],
            )
            split_docs = text_splitter.split_documents(documents)

            # 4. FAISS Vector Store 생성 및 저장
            self.vector_store = FAISS.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
            )
            self.vector_store.save_local(
                folder_path=self.folder_path,
                index_name=self.index_name
            )

            return f"성공적으로 {len(split_docs)}개의 청크가 정제되어 FAISS 인덱스에 저장되었습니다."
        except Exception as e:
            return f"FAISS VectorDB 적재 중 오류 발생: {str(e)}"

    def _get_vector_store(self) -> FAISS:
        """기존 저장된 FAISS 인덱스를 로드하거나 인스턴스화합니다."""
        if self.vector_store is None:
            # 로컬 저장소에 저장된 인덱스가 존재하는지 확인
            faiss_file = os.path.join(self.folder_path, f"{self.index_name}.faiss")
            if os.path.exists(faiss_file):
                self.vector_store = FAISS.load_local(
                    folder_path=self.folder_path,
                    embeddings=self.embeddings,
                    index_name=self.index_name,
                    allow_dangerous_deserialization=True,  # 로컬 생성 pickle 파일 로드 허용
                )
            else:
                raise FileNotFoundError(
                    f"FAISS 인덱스 파일을 찾을 수 없습니다: {faiss_file}. 먼저 load_pdf_to_vector_db()를 실행해 주세요."
                )
        return self.vector_store

    def search_similar_documents(self, query: str, k: int = 3) -> str:
        """
        입력된 자연어 질의와 가장 관련성 높은 문서를 FAISS 인덱스에서 검색하여 반환합니다.

        :param query: 검색 질의어
        :param k: 검색할 최상위 문서 수
        :return: 정형화된 검색 결과 문자열
        """
        try:
            vector_store = self._get_vector_store()
            results = vector_store.similarity_search(query, k=k)

            if not results:
                return "관련된 금융/투자 정보를 찾을 수 없습니다."

            formatted_results = []
            for idx, doc in enumerate(results, start=1):
                page_info = doc.metadata.get("page", "미상")
                formatted_results.append(
                    f"[{idx}] (페이지: {page_info})\n내용: {doc.page_content.strip()}"
                )

            return "\n\n---\n\n".join(formatted_results)
        except Exception as e:
            return f"FAISS VectorDB 검색 실행 중 에러 발생: {str(e)}"