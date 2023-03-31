import time
import pandas as pd
import numpy as np
import pickle
import warnings
import click
from tabulate import tabulate
from pathlib import Path
from dotenv import dotenv_values
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential
import cohere
from cohere.responses import Embedding
from lib.obsidian_helpers import read_markdown_notes


# CONFIG
COHERE_API_KEY = dotenv_values(".env.secret")["COHERE_API_KEY"] or ""
DF_FILE = "_data/cohere_embeddings.csv"
CACHE_FILE = "_data/cohere_query_cache.pkl"
OBSIDIAN_VAULT_PATH = dotenv_values(".env.secret")["OBSIDIAN_VAULT_PATH"]

co = cohere.Client(api_key=COHERE_API_KEY)

EMBEDDING_CTX_LENGTH = 4096
EMBEDDING_ENCODING = "cl100k_base"


def num_tokens_from_string(string: str, encoding_name=EMBEDDING_ENCODING) -> int:
    """Returns the number of tokens in a text string."""
    enc = tiktoken.get_encoding(encoding_name)
    num_tokens = len(enc.encode(string))
    return num_tokens


def truncate_text(
    text: str, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH
) -> str:
    """Truncate a string to have `max_tokens` according to the given encoding."""
    enc = tiktoken.get_encoding(encoding_name)
    encoded = enc.encode(text)
    if len(encoded) > max_tokens:
        return enc.decode(encoded[:max_tokens])
    else:
        return text


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.float64:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embedding(block: str) -> Embedding:
    return co.embed([block]).embeddings[0]


def build_embeddings(root: Path, df_file=DF_FILE):
    # get all notes
    notes = read_markdown_notes(root)
    # Embed and save
    df = embed(notes)
    click.echo("Saving df.")
    df.to_csv(df_file)


def read_df_file(df_file=DF_FILE) -> pd.DataFrame:
    # Util needed since some of my multi-index entries are empty strings.
    df = pd.read_csv(df_file, header=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(
        [tuple(["" if y.find("Unnamed") == 0 else y for y in x]) for x in df.columns]
    )
    return df


def update_embeddings(root: Path, df_file=DF_FILE):
    # get all notes
    notes = read_markdown_notes(root)

    # read df
    df = read_df_file(df_file)
    new_df = embed(notes)

    df = pd.concat([df, new_df], axis=1)
    click.echo("Saving df.")
    df.to_csv(df_file)


def embed(notes: dict[tuple[str, str], str]) -> pd.DataFrame:
    # Embeds the notes into openAI and returns a dataframe containing the vectors.
    res = {}
    showfunc = lambda n: f"{n[0][0]} {n[0][1]}" if n else ""
    with click.progressbar(notes.items(), item_show_func=showfunc) as note_items:
        for (note, section), text in note_items:
            block = section + ". " + text
            n = num_tokens_from_string(block)
            # Truncate if too long
            if n > EMBEDDING_CTX_LENGTH:
                warnings.warn(f"{note} {section} exceeded token limit. Truncating.")
                block = truncate_text(block)
            try:
                embedding = get_embedding(block)
            except Exception as e:
                print(f"Error for {note} {section}", e)
                continue
            res[(note, section)] = embedding
            time.sleep(0.1)
    df = pd.DataFrame(res)
    return df


def query_embeddings(qstr: str, df_file=DF_FILE) -> pd.Series:
    # Given a query string, compare against the embedded notes
    # and return them in order of similarity.
    try:
        df = read_df_file(df_file)
    except FileNotFoundError:
        raise click.ClickException(
            "Could not find database, please run with --build flag"
        )

    # Make cache if it doesn't exist
    try:
        cache = pickle.load(open(CACHE_FILE, "rb"))
    except (OSError, IOError):
        cache = {}

    # Return from cache if it's there else hit API.
    if qstr in cache:
        qvec = cache[qstr]
    else:
        qvec = get_embedding(qstr)
        cache[qstr] = qvec
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)

    # Return notes sorted by similarity
    cos_sim = np.apply_along_axis(lambda x: cosine_similarity(x, qvec), axis=0, arr=df)
    results = pd.Series(cos_sim, index=df.columns).sort_values(ascending=False)
    return results


def find_near_unconnected():
    # Based on the embedding vectors, find notes that are near each other but not connected.
    # These are prime candidates for linkage.
    pass


def present_results(results: pd.Series, root: str) -> str:
    # Format the results into a nice table
    resdf = results.reset_index()
    resdf.columns = ["Note", "Section", "Similarity"]
    resdf["Similarity"] = resdf["Similarity"].round(3)
    resdf = resdf.rename_axis("id", axis=0)

    return tabulate(resdf, headers="keys", tablefmt="psql")


@click.command()
@click.argument("query", required=False)
@click.option("--n", default=10, help="Number of responses to put in ")
@click.option("--build", is_flag=True, help="Recomputes all the embeddings.")
@click.option("--update", is_flag=True, help="Computes embeddings for new notes.")
@click.option(
    "--root",
    type=click.Path(exists=True),
    default=OBSIDIAN_VAULT_PATH,
    help="Path to the root of the vault.",
)
def main(query, n, build, update, root):
    """AI augmented search Obsidian notes."""
    if build:
        click.echo("Building embeddings...")
        build_embeddings(root)
    elif update:
        click.echo("Updating embedings...")
        update_embeddings(root)
    if query:
        results = query_embeddings(query)
        results_sub = results.iloc[:n]
        click.echo(present_results(results_sub, root))
        click.echo()
    else:
        click.echo("No query provided.")


if __name__ == "__main__":
    main()
