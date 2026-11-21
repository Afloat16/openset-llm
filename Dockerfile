# Engine image: a portable build (every x86-64 kernel variant; the best one for the host CPU is picked at start).
# The model is NOT in the image; it is downloaded from Hugging Face into a volume on first start.
#
#   docker build -t openset-llm .
#   docker run --rm -p 127.0.0.1:8080:8080 -v openset-llm-models:/data/models openset-llm
#   curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
#        -d '{"messages":[{"role":"user","content":"Hello"}]}'
#
# Give the container at least 8 GB of memory (Docker Desktop's default VM is often smaller).
FROM ubuntu:24.04 AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY patches/ patches/
COPY kernels/ kernels/
RUN git clone -q https://github.com/PrismML-Eng/llama.cpp llama.cpp \
 && git -C llama.cpp checkout -q e3de715222b2667c54a267441159a5729f11155e \
 && sh patches/apply.sh llama.cpp \
 && cmake -S llama.cpp -B llama.cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON \
      -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN' \
 && cmake --build llama.cpp/build -j"$(nproc)" \
 && mkdir -p /out/bin && cp -a llama.cpp/build/bin/llama-server llama.cpp/build/bin/*.so* /out/bin/ \
 && cp -a llama.cpp/gguf-py /out/gguf-py

FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl python3 python3-numpy python3-yaml libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/bin /opt/openset-llm/bin
COPY --from=build /out/gguf-py /opt/openset-llm/llama.cpp/gguf-py
COPY install.sh serve /opt/openset-llm/app/
COPY patches/ /opt/openset-llm/app/patches/
COPY kernels/ /opt/openset-llm/app/kernels/
COPY scripts/ /opt/openset-llm/app/scripts/
COPY data/ /opt/openset-llm/app/data/
COPY docker/entrypoint.sh /opt/openset-llm/app/entrypoint.sh
RUN chmod +x /opt/openset-llm/app/install.sh /opt/openset-llm/app/serve /opt/openset-llm/app/entrypoint.sh /opt/openset-llm/app/scripts/*.sh
ENV OPENSET_LLM_HOME=/opt/openset-llm
VOLUME /data/models
EXPOSE 8080
ENTRYPOINT ["/opt/openset-llm/app/entrypoint.sh"]
