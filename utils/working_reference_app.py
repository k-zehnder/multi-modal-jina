import os
import pathlib
import shutil
from jina import Flow, Document, DocumentArray
import matplotlib.pyplot as plt
from jina import Executor, Flow, requests
from jina.logging.logger import JinaLogger
from jina.types.request import Request
import torch
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer
from typing import Optional, Dict, List, Sequence
import torchvision


def plot_search_results(resp: Request):
    for doc in resp.docs:
        print(f'Query text: {doc.text}')
        print(f'Matches:')
        print('-' * 10)
        # doc.matches[:3].plot_image_sprites()
        print([d.uri for d in doc.matches[:3]])

class SimpleIndexer(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(os.path.join(self.workspace, 'index.db'))
        self._index = DocumentArray(
            storage='sqlite',
            config={
                'connection': os.path.join(self.workspace, 'index.db'),
                'table_name': 'clip',
            },
        )

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        self._index.extend(docs)

    @requests(on='/search')
    def search(self, docs: DocumentArray, **kwargs):
        docs.match(self._index)


class CLIPImageEncoder(Executor):
    """Encode image into embeddings using the CLIP model."""

    def __init__(
        self,
        pretrained_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

        self.device = device
        self.preprocessor = CLIPFeatureExtractor.from_pretrained(pretrained_model_name)
        self.model = CLIPModel.from_pretrained(
            pretrained_model_name
        )  # load the pretrained clip model from the transformer library

        self.model.to(
            self.device
        ).eval()  # we want to do only inference so we put the model in eval mode

    @requests
    @torch.inference_mode()  # we don't want to keep track of the gradient during inference
    def encode(self, docs: DocumentArray, parameters: dict, **kwargs):

        for batch_docs in docs.batch(
            batch_size=self.batch_size
        ):  # we want to compute the embedding by batch of size batch_size
            tensor = self._generate_input_features(
                batch_docs
            )  # Transformation from raw images to torch tensor
            batch_docs.embeddings = (
                self.model.get_image_features(**tensor).cpu().numpy()
            )  # we compute the embeddings and store it directly in the DocumentArray

    def _generate_input_features(self, docs: DocumentArray):
        docs.apply(lambda d: d.load_uri_to_image_tensor())
        input_features = self.preprocessor(
            images=[d.tensor for d in docs],
            return_tensors="pt",
        )
        input_features = {
            k: v.to(torch.device(self.device)) for k, v in input_features.items()
        }
        return input_features

class CLIPTextEncoder(Executor):
    """Encode text into embeddings using the CLIP model."""

    def __init__(
        self,
        encode_text=True,
        pretrained_model_name: str = 'openai/clip-vit-base-patch32',
        device: str = 'cpu',
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.device = device

        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name
        )  # load the tokenizer from the transformer library

        self.model = CLIPModel.from_pretrained(
            pretrained_model_name
        )  # load the pretrained clip model from the transformer library

        self.model.eval().to(
            device
        )  # we want to do only inference so we put the model in eval mode

    @requests
    @torch.inference_mode()  # we don't want to keep track of the gradient during inference
    def encode(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):

        for docs_batch in docs.batch(
            batch_size=self.batch_size
        ):  # we want to compute the embedding by batch of size batch_size
            input_tokens = self._generate_input_tokens(
                docs_batch.texts
            )  # Transformation from raw texts to torch tensor
            docs_batch.embeddings = (
                self.model.get_text_features(**input_tokens).cpu().numpy()
            )  # we compute the embeddings and store it directly in the DocumentArray

    def _generate_input_tokens(self, texts: Sequence[str]):

        input_tokens = self.tokenizer(
            texts,
            max_length=77,
            padding='longest',
            truncation=True,
            return_tensors='pt',
        )
        input_tokens = {k: v.to(self.device) for k, v in input_tokens.items()}
        return input_tokens

images = DocumentArray.from_files("/Users/peppermint/Desktop/multi-modal-jina/data/images/*.jpg", num=50).apply( # num=200
            lambda d: d.load_uri_to_image_tensor()
            .load_uri_to_image_tensor(200, 200)  # load
            .set_image_tensor_normalization()  # normalize color
            .set_image_tensor_channel_axis(-1, 0)  # switch color axis for the PyTorch model later
)
current_dir = pathlib.Path(__file__).parent.resolve()
if os.path.exists(os.path.join(current_dir, "workspace")):
    print("[INFO] removing existing workspace...")
    shutil.rmtree(os.path.join(current_dir, "workspace"))

#lsof -n -i4TCP:8080
# kill -9 21023
#PEXELS API=563492ad6f9170000100000134462d29bc3e4e7e87bc2d4f88929fa1

flow_index = (
    Flow(port=12345)
    .add(uses=CLIPImageEncoder, name='encoder', uses_with={'device': "cpu"})
    .add(uses=SimpleIndexer, name='indexer', workspace='workspace')
)
# flow_index.protocol='http'

with flow_index:
    flow_index.post(on='/index', inputs=images, on_done=print, return_results=True)

flow_search = (
    Flow(port=12345)
    .add(uses=CLIPTextEncoder, name='encoder', uses_with={'device': "cpu"})
    .add(uses=SimpleIndexer, name='indexer', workspace='workspace')
)
# flow_index.protocol='http'

# with flow_search:
#     flow_search.block()

with flow_search:
    resp = flow_search.search(
        inputs=DocumentArray(
            [
                Document(text='stock market'),
                Document(text='cards'),
            ]
        ),
        on_done=plot_search_results, return_results=True
    )

# with flow_search:
#     flow_search.protocol = "http"
#     flow_search.port_expose = 12345
#     flow_search.block()