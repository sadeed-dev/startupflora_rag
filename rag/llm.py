"""LangChain model adapters with exact-text vector caching and bounded inputs."""
# This file connects to AI models, caches text vectors, and checks input limits.

import json

# Converts Python dictionaries/lists into JSON text.
from typing import TypeVar

# Lets us describe a flexible type for a function's return value.
from langchain_core.embeddings import Embeddings

# Base class defining the methods an embedding implementation should provide.
from langchain_core.messages import SystemMessage

# Creates a message containing instructions for the AI.
from langchain_core.prompts import ChatPromptTemplate

# Builds AI messages using fixed instructions and placeholders.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ChatOpenAI: creates a client for generating AI answers.
# OpenAIEmbeddings: creates a client for converting text into vectors.
from pydantic import BaseModel

# Used to define the fields and types expected in structured answers.
from .text import digest, token_count
from .timing import stage
from .usage import EmbeddingUsageClient, TokenUsageCallback

# Imports two helpers from your project's text.py file.
# digest(): creates an identifier from text.
# token_count(): counts tokens in text.
# The "." means text.py belongs to the same Python package.

T = TypeVar("T", bound=BaseModel)
# T represents any class derived from Pydantic's BaseModel.
# Used later to describe that structured() returns the supplied schema's type.
# This is a type hint; it does not perform validation itself.


class ModelFailure(Exception):
    # Create a custom exception type for model-related failures.

    """Signal an incomplete, refused or invalid model result without exposing provider data."""
    # Example use:
    # raise ModelFailure("Incomplete embedding batch")
    # Raising this error stops normal execution unless another part catches it.


class CachedEmbeddings(Embeddings):
    # Create our own embedding implementation using LangChain's base class.
    # It checks saved vectors before asking the provider to create new ones.

    """Implement the LangChain embedding interface while caching by text/model/dimensions."""

    def __init__(self, store, provider, signature):
        # Runs when a CachedEmbeddings object is created.
        # self: this particular object.
        # store: object that reads and saves cached vectors.
        # provider: embedding model/client.
        # signature: identifies the embedding model and vector size.

        """Keep one provider and a shared on-disk vector cache across CLI/API restarts."""

        self.store, self.provider, self.signature = store, provider, signature
        # Save all three arguments inside this object.
        # Equivalent to:
        # self.store = store
        # self.provider = provider
        # self.signature = signature

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Takes a list of strings and returns a list of vectors.
        # Each vector is a list of decimal numbers.
        # Example input: ["Hello", "Learn Python"]
        # Example output: [[0.1, 0.2, ...], [0.3, 0.4, ...]]

        """Deduplicate exact inputs and embed only missing texts in small batches."""

        keys = [digest(self.signature + "\n" + text) for text in texts]
        # For every text:
        # 1. Combine the model signature, a newline, and the text.
        # 2. Pass that combined string to digest().
        # 3. Keep the resulting cache key.
        #
        # The signature separates caches for different models/vector sizes.
        # Identical text with the same signature produces the same key.

        vectors = {key: self.store.cached_vector(key) for key in dict.fromkeys(keys)}
        # dict.fromkeys(keys) removes duplicate keys while keeping their order.
        # cached_vector(key) reads the saved vector for each unique key.
        # If no saved vector exists, this code expects None.
        #
        # Example:
        # vectors = {
        #     "key_hello": [0.1, 0.2, ...],
        #     "key_python": None,
        # }

        missing = {}
        # Empty dictionary for texts that still need vectors.

        for key, text in zip(keys, texts):
            # zip() pairs each key with its original text.

            if vectors[key] is None:
                # Check whether this text's vector is missing from the cache.

                missing[key] = text
                # Save the missing text under its key.
                # Duplicate keys occupy only one dictionary entry,
                # so repeated identical text is processed once.

        missing_keys = list(missing)
        # Convert the missing dictionary's keys into a list.

        for start in range(0, len(missing_keys), 32):
            # Process missing texts in groups of up to 32.
            # start takes values 0, 32, 64, ...
            # If nothing is missing, this loop does not run.

            batch = missing_keys[start : start + 32]
            # Select up to 32 keys for this group.
            # Example: missing_keys[0:32] selects the first 32 keys.

            inputs = [missing[key] for key in batch]
            # Get the actual text belonging to each selected key.

            if any(token_count(text) > 7500 for text in inputs):
                # Count tokens for each text in this group.
                # any() returns True if at least one text exceeds 7,500 tokens.
                # This is the limit chosen by this code.

                raise ModelFailure(
                    "Embedding input too long; reduce CHUNK_SIZE or clean the source title"
                )
                # Stop because a text is too large.
                # Smaller chunks can help keep each input within this limit.

            computed = self.provider.embed_documents(inputs)
            # Send the missing texts to the embedding provider.
            # Receive one vector for each text.

            if len(computed) != len(batch):
                # Compare the number of returned vectors with the number sent.
                # Example: 10 texts must produce 10 vectors.

                raise ModelFailure("Incomplete embedding batch")
                # Stop if the provider returned an unexpected number of vectors.

            expected = int(self.signature.rsplit(":", 1)[1])
            # Extract the expected vector size from the signature.
            #
            # Example signature: "text-embedding-3-large:3072"
            # rsplit(":", 1) -> ["text-embedding-3-large", "3072"]
            # [1]           -> "3072"
            # int(...)      -> 3072

            if any(len(vector) != expected for vector in computed):
                # Check that every vector contains the expected number of values.
                # For expected=3072, every vector must have 3,072 values.

                raise ModelFailure("Embedding dimension mismatch")
                # Stop if any vector has the wrong size.

            for key, vector in zip(batch, computed):
                # Pair each cache key with its newly created vector.

                vectors[key] = vector
                # Add the new vector to the dictionary held in memory.

            self.store.cache_vectors(list(zip(batch, computed)))
            # Create pairs such as:
            # [("key_1", vector_1), ("key_2", vector_2)]
            # Save these pairs through the store for future reuse.

        return [vectors[key] for key in keys]
        # Return one vector for every original input position.
        # Preserve the original order, including repeated texts.

    def embed_query(self, text: str) -> list[float]:
        # Takes one string, usually a user's search question.
        # Returns one vector: a list of decimal numbers.

        """Use the same embedding space and cache for user queries as for indexed text."""

        return self.embed_documents([text])[0]
        # [text] wraps the question in a one-item list.
        # embed_documents([text]) returns a list containing one vector.
        # [0] gets that first vector.
        # This uses the same provider and cache as document embeddings.


class Models:
    # Groups model setup, input checks, structured output, and cleanup.

    """Centralize provider settings and strict structured output through LangChain."""

    def __init__(self, settings, store, chat=None, utility=None, embeddings=None):
        # Runs when a Models object is created.
        # settings: your configuration object.
        # store: your embedding cache storage object.
        # chat, utility, embeddings: optional supplied model objects.
        # None means no model object was supplied for that argument.

        """Allow injected LangChain models/transports for deterministic offline tests."""

        self.s = settings
        # Save configuration so other methods can access it.

        self.signature = f"{settings.embedding_model}:{settings.dimensions}"
        # Build an identifier from the embedding model and vector size.
        # Example: "text-embedding-3-large:3072"

        args = dict(
            # Create settings shared by both chat model clients.

            api_key=settings.openai_key,
            # API key used to authenticate requests.

            timeout=settings.model_timeout_seconds,
            # Allow slower generations without changing the embedding timeout.

            max_retries=0 if settings.fast_mode or settings.minimum_calls else 1,
            # Fast requests must not spend a second timeout on retries.

            temperature=0,
            # Request less random output.
            # This does not guarantee identical answers every time.

            max_tokens=settings.max_output_tokens if settings.minimum_calls else settings.answer_max_tokens,
            # Default maximum number of output tokens.

            store=False,
            # Request that the provider not store the generated response.
            # This setting is separate from your local embedding cache.

            use_responses_api=False,
            # Configure this client to use the Chat Completions API.
        )

        chat_class = ChatOpenAI
        if settings.chat_provider == "gemini":
            from .gemini import GEMINI_BASE_URL, GeminiChat

            chat_class = GeminiChat
            args.pop("store", None)
            args.update(api_key=settings.gemini_key, base_url=GEMINI_BASE_URL,
                        reasoning_effort=settings.gemini_reasoning_effort)
        self.chat = chat or chat_class(model=settings.effective_chat_model,
            callbacks=[TokenUsageCallback(settings.chat_provider, settings.effective_chat_model)], **args)
        # Use chat if a truthy model object was supplied.
        # Otherwise create a ChatOpenAI client with the configured chat model.
        # **args passes all dictionary entries as named arguments.

        self.utility = utility or chat_class(model=settings.effective_utility_model,
            callbacks=[TokenUsageCallback(settings.chat_provider, settings.effective_utility_model)], **args)
        # Use the supplied utility model or create one.
        # structured() below uses this model for structured answers.

        provider = embeddings or OpenAIEmbeddings(
            # Use supplied embeddings or create an OpenAI embedding client.

            api_key=settings.openai_key,
            base_url="https://api.openai.com/v1",
            # API key for embedding requests.

            model=settings.embedding_model,
            # Name of the embedding model.

            dimensions=settings.dimensions,
            # Requested number of values in each vector.

            request_timeout=5 if settings.fast_mode else 25,
            # Leave time for generation after query embedding.

            max_retries=0 if settings.fast_mode or settings.minimum_calls else 1,
            # Allow one retry for a retryable failure.

            # Chunks are already bounded. This avoids another tokenizer/download
            # dependency inside the provider and preserves exact cache text.
            check_embedding_ctx_length=False,
            # Disable the provider's automatic input-length checking/splitting.
            # embed_documents() above checks each text's length itself.
        )

        if embeddings is None:
            provider.client = EmbeddingUsageClient(provider.client, settings.embedding_model)
        self.embeddings = CachedEmbeddings(store, provider, self.signature)
        # Wrap the embedding provider with our cache logic.
        # Calling self.embeddings can now reuse saved vectors.

    def guard(self, messages, tools=None):
        """Estimate the whole serialized input, including schema and protocol overhead."""
        from langchain_core.messages import convert_to_openai_messages
        from langchain_core.utils.function_calling import convert_to_openai_tool

        payload = {
            "messages": convert_to_openai_messages(messages),
            "tools": [convert_to_openai_tool(t, strict=True) for t in tools or []],
            "tool_choice": "Answer", "parallel_tool_calls": False,
            "model": self.s.effective_chat_model,
        }
        estimated = token_count(json.dumps(payload, ensure_ascii=False)) + 64
        if estimated > self.s.model_input_tokens:
            raise ModelFailure("Input budget exceeded; reduce retrieval or history budgets")
        return estimated

    def structured(self, schema: type[T], instructions: str, payload: dict, max_tokens=1000) -> T:
        # Ask the utility model for an answer matching a Pydantic schema.
        # schema: Pydantic model class describing the expected answer.
        # instructions: what the AI should do.
        # payload: input data to process.
        # max_tokens: output limit; defaults to 1,000.
        # -> T: the returned object has the supplied schema's type.

        """Use a typed LangChain output parser and reject refusals/truncated generations."""

        prompt = ChatPromptTemplate.from_messages(
            # Build a prompt containing two messages.

            [SystemMessage(content=instructions), ("human", "{payload}")]
            # Message 1: system instructions.
            # Message 2: human message with a {payload} placeholder.
        )

        messages = prompt.invoke({"payload": json.dumps(payload, ensure_ascii=False)}).to_messages()
        # 1. Convert the input dictionary to JSON.
        # 2. Fill the prompt's {payload} placeholder with that JSON.
        # 3. Convert the filled prompt into a list of messages.

        self.guard(messages, [schema])
        # Check the input size, including the schema definition.
        # If the input is too large, guard() raises an error.

        # Clone configuration, not the underlying HTTP client. Binding a model
        # before with_structured_output can lose invocation kwargs in adapters.
        runner = self.utility.model_copy(update={"max_tokens": max_tokens}).with_structured_output(
            # Copy the utility model configuration with this request's output limit.
            # Then configure structured output on that copy.

            schema,
            # Use this Pydantic class as the required answer format.

            method="json_schema",
            # Request structured output using JSON Schema.

            strict=True,
            # Request strict compliance with the schema.

            include_raw=True
            # Include the original response along with the parsed result/errors.
        )

        with stage(f"model_{schema.__name__.lower()}"):
            output = runner.invoke(messages)
        # Send the messages to the model.
        # The result includes raw, parsed, and parsing_error entries.

        raw = output.get("raw")
        # Get the original response.
        # .get() returns None if the key is absent.

        if raw and raw.response_metadata.get("finish_reason") in (
            # If a raw response exists, inspect why generation finished.

            "length",
            # Generation stopped because it reached its output limit.

            "content_filter",
            # Generation stopped because of content filtering.
        ):
            raise ModelFailure("Incomplete structured model response")
            # Reject these responses because they may be incomplete.

        if output.get("parsing_error") or output.get("parsed") is None:
            # Reject the result if:
            # 1. A parsing error occurred, OR
            # 2. No parsed object was returned.

            raise ModelFailure("Model refused or returned an invalid structured result")
            # Stop instead of returning an unusable answer.

        return output["parsed"]
        # Return the successfully parsed Pydantic object.

    def close(self):
        # Close synchronous API clients when the application shuts down.

        """Close synchronous provider clients when the CLI or API shuts down."""

        for provider in (self.chat, self.utility):
            # Process the chat model and utility model.

            client = getattr(provider, "root_client", None)
            # Look for the provider's root_client attribute.
            # Use None if it does not exist.

            if client:
                # If a client was found...

                client.close()
                # ...close it to release its connections/resources.

        embedding_client = getattr(
            # Look for the underlying embedding client.

            getattr(self.embeddings.provider, "client", None), "_client", None
            # First get self.embeddings.provider.client.
            # Then get that object's _client.
            # Use None if either attribute is missing.
        )

        if embedding_client:
            # If the underlying embedding client was found...

            embedding_client.close()
            # ...close it too.
