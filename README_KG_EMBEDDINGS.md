# PreDiViD: reconstituição das observações e novos embeddings (KG -> Prediction)

Este é um **overlay**: extraia o ZIP diretamente na raiz de `DDPF-Health-Risks`. Ele **não altera** os notebooks, a ontologia ou o banco Neo4j. Os scripts Python consultam uma cópia restaurada e isolada do dump `BioPropaPhenKG-COVIDvsMPOX.dump` em modo **READ**.

## Descoberta nos notebooks originais

O notebook `Experiments/Detection/Detection Module-COVIDvsMPOX.ipynb`, células 57–59, gerou os seis CSVs históricos a partir do Neo4j usando:

```cypher
MATCH (n:Country)<-[r:hasPresence]-(c)
WHERE toInteger(r.intensity) >= $threshold AND r.source = $source
RETURN n.wkgs_nameEn AS System_Name, n.id, c.name, c.id,
       r.intensity AS intensity
```

Os limiares usados são exatamente:

| `r.source` | `intensity` mínima | CSV histórico |
|---|---:|---|
| Journal_COVID | 5 | covid_observations_journal.csv |
| Medical_COVID | 8 | covid_observations_medical.csv |
| Social_COVID | 10 | covid_observations_social.csv |
| Journal_Monkeypox | 5 | monkeypox_observations_journal.csv |
| Medical_Monkeypox | 3 | monkeypox_observations_medical.csv |
| Social_Monkeypox | 10 | monkeypox_observations_social.csv |

O notebook `Prediction Module - COVID vs Monkeypox Observation.ipynb` depois ordena e limita os conceitos de cada observação aos **25** mais intensos e faz BoW binário + K-Means. O arquivo `scripts/realdata/replay_bow.py` do pacote de reprodução anterior deve continuar disponível para o replay histórico.

**Condição:** as relações `hasPresence` com `source` e `intensity` precisam existir no dump efetivamente restaurado. A existência de `isReported` sozinho **não basta**. Consulte primeiro com `--inspect`.

**Unidade de observação recuperada:** país × fonte/corpus. Os CSVs não recuperam textos originais, IDs de artigos ou segmentos de GATE e não servem por si só como gold standard para links do Story Forest. Datas, quando existentes, são metadados dos relacionamentos, não datas de evento verificadas.

## 1. Instalação no servidor

No repositório com `.venv` ativo:

```bash
python -m pip install neo4j python-dotenv numpy scipy scikit-learn
```

Verifique que o dump foi restaurado em uma instância Neo4j isolada e que `.env` contém:

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=SEU_SEGREDO
NEO4J_DATABASE=neo4j
OLLAMA_BASE_URL=http://localhost:11434
```

`NEO4J_USER` também é aceito. O script **não restaura** automaticamente o dump nem altera o Neo4j. Caso o Ollama esteja no Windows e o script no Linux, `localhost` no Linux **não** se refere ao Windows: use um servidor Ollama acessível ao Linux, de forma segura.

## 2. Inspecionar o KG (consulta somente-leitura)

```bash
python scripts/realdata/recover_kg_observations.py --inspect
```

Procure `hasPresence_in_type_catalog: true` e amostras com campos `source` entre as seis fontes listadas acima. Se vierem vazias, verifique a restauração e o esquema antes de prosseguir.

## 3. Reconstruir os seis CSVs do notebook

```bash
python scripts/realdata/recover_kg_observations.py
```

O script examina somente `hasPresence` entre conceitos e países, aplica as **seis condições originais** em uma consulta, faz o agrupamento em SQLite temporário (para não carregar todas as arestas na RAM), gera as seis tabelas em `data/neo4j/`, e gera:

```
outputs/realdata/reconstructed_observations.jsonl
outputs/realdata/reconstruction_manifest.json
```

Os novos CSVs têm as colunas originais `System_Name`, `n.id`, `c.name`, `c.id`, `intensity` e uma coluna adicional `relation_dates`, preservando o alinhamento com a representação histórica. Valores `[c.name, c.id, intensity]` são listas serializadas como no notebook original. A ordenação de empates é determinística e pode diferir da execução histórica; os resultados são **reconstituição/replay**, não garantia de identidade bit a bit com CSVs antigos.

**Smoke test:** usar `--max-relations 2000 --out outputs/realdata/smoke` produzirá um subconjunto, marcado como tal. Não use esses resultados como avaliação científica. O script nunca sobrescreve saídas existentes. Para uma nova reconstrução, escolha outro `--out` e passe o novo `--input` nas etapas seguintes.

## 4. Primeiro benchmark histórico: BoW

Se `scripts/realdata/replay_bow.py` já estiver instalado pelo pacote anterior:

```bash
python scripts/realdata/replay_bow.py --check
python scripts/realdata/replay_bow.py
```

Em seguida compare com variantes sem t-SNE intermediário na etapa 6. Note que algumas células do notebook histórico usavam t-SNE **antes** do K-Means, o que não é aconselhável como avaliação principal.

## 5. Novas matrizes com Ollama

São **modelos de embeddings textuais especializados**. Não use `qwen3:4b` como substituto automático de `qwen3-embedding:4b`.

```bash
ollama pull qwen3-embedding:0.6b
ollama pull embeddinggemma
ollama pull nomic-embed-text
# Se houver RAM/VRAM e tempo suficientes:
ollama pull qwen3-embedding:4b
```

Modelos em lote (padrão = 8 observações/chamada, progressão no terminal):

```bash
python scripts/realdata/embed_kg_observations.py \
  --models qwen3-embedding:0.6b embeddinggemma nomic-embed-text \
  --features concepts --batch-size 8
```

O script produz, para cada modelo:

```
outputs/realdata/embeddings/<modelo>/concepts/embeddings.npy
outputs/realdata/embeddings/<modelo>/concepts/observations.csv
outputs/realdata/embeddings/<modelo>/concepts/manifest.json
```

`embeddings.npy` é `float32` com uma linha por `observation_id`; o CSV de alinhamento salva os IDs e rótulos para **avaliação apenas**. O texto enviado ao Ollama contém os nomes dos conceitos, **não** os campos `source` ou `disease_label`.

O processo tem checkpoint em `progress.json`; se cair a ligação SSH ou a sessão for interrompida, rerodar o mesmo comando recupera as linhas concluídas, verificando o hash do dataset. A saída de um modelo concluído também é reutilizada, sem sobrescrever.

### Ablações espaço-temporais

Comparar concept-only com condições explícitas, sem confundi-las com resultados antigos:

```bash
python scripts/realdata/embed_kg_observations.py --models qwen3-embedding:0.6b --features concepts_country
python scripts/realdata/embed_kg_observations.py --models qwen3-embedding:0.6b --features concepts_time
python scripts/realdata/embed_kg_observations.py --models qwen3-embedding:0.6b --features concepts_country_time
```

A versão temporal só introduz datas quando `r.date` foi encontrado. Ela não fabrica datas. A análise com datas é especialmente vulnerável à confusão porque os corpora COVID e Mpox foram recolhidos em períodos diferentes.

### Testar vazamento do nome da doença

```bash
python scripts/realdata/embed_kg_observations.py \
  --models qwen3-embedding:0.6b embeddinggemma nomic-embed-text \
  --features concepts --mask-disease-terms
```

A ablação suprime conceitos cujo nome menciona explicitamente COVID-19, SARS-CoV-2, coronavirus, monkeypox, mpox ou orthopoxvirus, sem excluir países/observações. Os embeddings são salvos em `concepts_masked` para não sobrescrever as matrizes integrais. Trata-se de uma **ablação**, não de uma forma de garantir ausência total de viés.

## 6. Comparar embeddings e clusters (múltiplas seeds)

```bash
python scripts/realdata/compare_kg_embeddings.py
cat outputs/realdata/embedding_benchmark/results.csv
```

Também gera `results.json` com resultados individuais por seed. K-Means trabalha nas matrizes originais, não numa projeção t-SNE; calcula ARI, NMI, F1 macro (depois de alinhar rótulos dos clusters), acurácia e acurácia balanceada.

**As métricas não avaliam qualidade de ligação de eventos.** São métricas de separação entre corpora COVID e Mpox, que também podem refletir período, seleção e vocabulário. Use validação externa independente para o novo artigo. Para Story Forest, somente documentos com proveniência e datas justificam classificação de links como `documented_continuation`.

## 7. Manter a execução em tmux

```bash
cd /data/dataRapide/gabriel/DDPF-Health-Risks
tmux new -s kg-embeddings
# Dentro do tmux:
mkdir -p logs
set -o pipefail
bash scripts/realdata/run_kg_embeddings_tmux.sh 2>&1 | tee logs/kg_embeddings.log
```

Para soltar a sessão: `Ctrl+B`, depois `D`. Para voltar: `tmux attach -t kg-embeddings`.

O script verifica Neo4j e modelos locais. Depois reconstitui CSVs e executa as matrizes e benchmark sequencialmente. Os arquivos manifest permitem identificar os inputs para a publicação.

## 8. Testes locais sem Neo4j nem Ollama

```bash
python -m unittest discover -s tests/realdata -p 'test_embedding_recovery.py' -v
```

## 9. Experimentos estruturais com KG e novas contribuições

Este pacote trabalha no nível de **observações agregadas**, adequado para a reexecução do primeiro Prediction. **FastRP, Node2Vec e GraphSAGE são embeddings estruturais de nós ou grafos**, não modelos de texto do Ollama. Regenerá-los exigirá uma segunda etapa com projeções Neo4j GDS ou um pipeline de grafo próprio, com controle de memória e versão. Não tente projetar os ~14M nós / 54M arestas inteiros automaticamente na RAM do servidor sem dimensionamento prévio. O objetivo do próximo módulo será recuperar `CUI` e `Country` com suas arestas, produzir embeddings estruturais e agregá-los por observação, sempre com IDs e verificações de integridade.
