from django.shortcuts import render
from rest_framework.viewsets import ViewSet
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import UploadSerializer, StringInputSerializer
from .utils import get_pdf_text, get_text_chunks, get_embeddings, insert_embedding
from .groq_llm import call_chain
from langchain_community.vectorstores.pgvector import PGVector
from .rag import rag_query

# ViewSets define the view behavior.
class UploadViewSet(ViewSet):
    serializer_class = UploadSerializer

    def list(self, request):
        return Response("GET API")

    def create(self, request):
        if 'file_uploaded' in request.FILES:
            file = request.FILES.get('file_uploaded')
            text = get_pdf_text(file)
            text_chunks = get_text_chunks(text)
            embedding_list = get_embeddings(text_chunks)
            insert_embedding(text_chunks, embedding_list)
            content_type = file.content_type
            response = "POST API and you have uploaded a {} file".format(content_type)
            print("File uploaded")
            return Response(response)
        
        serializer = StringInputSerializer(data=request.data)
        if serializer.is_valid():
            input_string = serializer.validated_data['input_string']
            retrieved_docs = rag_query(input_string)
            content = call_chain(retrieved_docs,input_string)
            response = content.replace('\n', ' ').strip()
            return Response({"results": response})