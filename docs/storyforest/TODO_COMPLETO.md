# PreDiViD — TODO completo para o artigo Story Forest + LLM + Ontologia

**Objetivo:** finalizar uma extensão científica do DDPF que constrói histórias epidemiológicas incrementalmente, usando BioPropaPhen/BioPropaPhenKG, LLMs e validação ontológica; publicar resultados auditáveis e um protótipo funcional no DDPF Studio.

**Revista principal:** *Discover Artificial Intelligence*, categoria a decidir entre **Research** e **Methodology** a partir dos resultados reais. Guia: https://link.springer.com/journal/44163/submission-guidelines

**Escopo científico a defender (sujeito a comprovação):** vinculação de eventos com grounding no KG e restrições ontológicas, formação incremental de árvores de histórias, e rastreabilidade de evidência. **Não** afirmar transmissão causal, predição de casos futuros, vigilância em tempo real validada ou utilidade clínica sem avaliação apropriada.

## Códigos de prioridade e critérios de parada

- **P0**: obrigatório para sustentar o artigo científico; fazer primeiro.
- **P1**: necessário para um pacote publicável e demonstrável, mas pode ser reduzido se o prazo apertar.
- **P2**: extensão opcional. Não bloquear a submissão por ela.
- **Gate A — Dados:** conseguir criar **eventos individuais com evidência e datas de disponibilização** a partir do snapshot e/ou das fontes, sem confundir clusters por doença com histórias de eventos.
- **Gate B — Validação:** existir **referência independente**, mesmo que parcial e silver, para pelo menos uma avaliação real de vínculos; silver negativo sintético serve apenas para testar restrições, não prova desempenho epidemiológico.
- **Gate C — Novidade:** ablação demonstra benefício específico de KG e/ou ontologia frente a LLM-only e baselines não-LLM; se não houver ganho, redefinir a contribuição antes de escrever conclusões fortes.

## 0. Congelar recursos, hipótese e escopo — P0

- [ ] Criar branch `storyforest-paper` no repositório DDPF, issue tracker e pasta `paper/`.
- [ ] Registrar versões/DOIs, datas de download, manifest e checksums dos dados: https://zenodo.org/records/10911980 (ontologia e BioPropaPhenKG); https://zenodo.org/records/10987743 (snapshot COVID/Mpox); verificar se o benchmark de notícias requer https://zenodo.org/records/10997359.
- [ ] Arquivar uma cópia imutável dos arquivos do Zenodo em `data/raw/` (fora do Git) e não sobrescrevê-los.
- [ ] Catalogar os artefatos dos experimentos de `Expert_Systems(1).zip`, `IJMI.zip` e notebooks de `Experiments/` para evitar reutilizar tabelas incompatíveis sem explicação.
- [ ] Escrever o problema em 3 frases: (i) as observações isoladas são difíceis de organizar; (ii) as relações são ambíguas e sujeitos a alucinação; (iii) a reconstrução incremental deve ser eficiente e rastreável.
- [ ] Fixar a ontologia, tipos de relação e o alvo da avaliação: **associação/continuação de acontecimentos**, não contágio causal automaticamente.
- [ ] Fixar perguntas RQ1 (qualidade dos vínculos), RQ2 (efeito da ontologia e KG), RQ3 (incrementalidade), RQ4 (fidelidade das evidências).
- [ ] Fixar hipóteses falsificáveis e baselines antes de olhar resultados finais.
- [ ] Fazer mini revisão atualizada de Story Forest/EventX, event threading, event relation extraction, event RAG, ontology-guided LLM, timeline/storyline construction, e explicações verificáveis. Criar tabela `paper/related_work_matrix.csv` com diferenças reais; não alegar pioneirismo sem evidência.
- [ ] Registrar o parecer recebido de *Expert Systems* e fazer mapa da resposta científica a cada crítica: **clustering sozinho não é novidade**.

## 1. Inspecionar a informação realmente recuperável do grafo — P0, GATE A

- [ ] Restaurar **cópia** do snapshot experimental no Neo4j compatível; registrar scripts, configuração e tempo da importação.
- [ ] Identificar labels/nós, tipos de relacionamento, datas, locais, IDs UMLS, doenças, tipos de fontes e propriedades de observações por consultas exploratórias read-only.
- [ ] Identificar se os registros são por **documento**, parágrafo, sentença ou já **agregados** por conceito/país/data.
- [ ] Determinar se cada observação possui ID persistente, URI da fonte, trecho, nome do arquivo, data do evento, data de publicação, data de ingestão e provenance.
- [ ] Conferir se as associações `hasPresence` e relações espaço-temporais fornecem *evidência textual individual*, não só totais agregados.
- [ ] Verificar campos que vazam o rótulo: `Journal_COVID`, `Journal_Monkeypox`, metadados WHO posteriores, ano como substituto trivial da doença. Não fornecer rótulos ou dados futuros ao método testado.
- [ ] Fazer consulta limitada/paginada que exporte apenas campos necessários, nunca carregar 13M nós/54M arestas inteiras na RAM.
- [ ] Exportar para `data/processed/events_raw.parquet` + `data/processed/provenance.parquet` com schema e checksums.
- [ ] Calcular perfil descritivo: quantidade de observações, granularidade, % com datas, fontes, documentos e locais recuperáveis, distribuição temporal e possíveis duplicatas.
- [ ] **Decisão Gate A:** se só existem menções agregadas, **não** chamar isso de eventos historicamente verificados. Recuperar uma amostra dos textos de origem ou restringir o trabalho a *evidence trajectories* com escopo e nome apropriados.

## 2. Criar um benchmark automático, honesto e não circular — P0, GATE B

- [ ] Definir schema imutável para `Event`: `event_id`, `description`, `timestamp_event`, `timestamp_publication`, `geo`, `concept_ids`, `source_ids`, `observation_ids`, `evidence_ids`.
- [ ] Definir labels de par: `same_event`, `documented_continuation`, `related_only`, `contradicted`, `unknown` e política de abstenção. Separar relação de árvore (`story_parent`) do tipo semântico do vínculo.
- [ ] Deduplicar notícias e versões republicadas antes da partição: a mesma fonte/documento nunca em treino e teste.
- [ ] Extrair casos positivos de **identidade de acontecimento comprovável** por IDs de reportagem/boletim, atualizações explicitamente referenciadas ou menções a incidentes compartilhados com evidência textual.
- [ ] Extrair vínculos de continuidade explicitamente documentada em boletins/linhas do tempo de fonte externa; registrar URL, trecho ou ID e a data em que a afirmação foi publicada.
- [ ] Tratar pares sem confirmação como `unknown`, não como negativos automáticos.
- [ ] Gerar contraexemplos sintéticos controlados de datas incompatíveis, tipos ontológicos incompatíveis ou locais impossíveis para **teste de restrições**; publicar separadamente os resultados sintéticos.
- [ ] Dedicar um *split* externo de teste totalmente reservado e cronológico quando possível. Evitar mesmo acontecimento, fonte duplicada ou informação futura em treino/validação/teste.
- [ ] Produzir cortes de controle com períodos sobrepostos e mesma doença para verificar se o sistema separa eventos e não só COVID-2019/20 de Mpox-2022.
- [ ] Medir cobertura do silver standard: percentuais positivos/contraditos/desconhecidos e vieses por fonte; especificar que faltam rótulos completos.
- [ ] Definir no início como medir **candidate retrieval recall**, porque o algoritmo não pode acertar vínculos que nunca recupera.
- [ ] Criar `data/benchmark/events.jsonl`, `pairs.jsonl`, `stories.jsonl`, `evidence.jsonl`, `splits.json` e documentação de geração.
- [ ] **Decisão Gate B:** se não houver vínculos reais com referência independente, restringir o artigo a consistência formal e desempenho incremental; **não** reportar Edge F1 de eventos reais como se estivesse validado.

## 3. Método novo: Ontology-Guided LLM Story Forest — P0

- [ ] Definir formalmente: observação ≠ evento ≠ história ≠ doença; eventos podem compartilhar conceitos sem serem o mesmo incidente.
- [ ] Definir tipos de ligação permitidos pela ontologia e regras de compatibilidade (domínio/range, precedência temporal, compatibilidade geográfica, identidade de entidades, fonte e evidências mínimas).
- [ ] Definir `EventBuilder`: agrupa menções do mesmo acontecimento com deduplicação, sem usar labels da avaliação.
- [ ] Definir `CandidateRetriever`: encontra top-k vizinhos via subgrafo BioPropaPhenKG, proximidade temporal e similaridade, com paginação e índices.
- [ ] Definir `LLMRelationClassifier`: julga apenas candidatos, em JSON validável, retornando tipo de relação, confidence/calibration, `evidence_ids`, `abstain`.
- [ ] Garantir que o LLM não possa introduzir fatos sem suporte: IDs de evidências existentes, JSON Schema, validação de output, retries controlados e logs de recusas.
- [ ] Definir `OntologyValidator`: regras determinísticas antes/depois do LLM, com motivos de rejeição explícitos e distinção entre `invalid` e `unknown`.
- [ ] Definir `StoryBuilder`: criar/merge/extend (e possível reparent) eventos, preservar orientação temporal; justificar escolha por **árvore** (um pai por evento) vs DAG (múltiplas relações reais).
- [ ] Implementar *abstenção* para vínculos sem evidência suficiente; não forçar uma árvore completa.
- [ ] Implementar `IncrementalUpdater` com checkpoint persistente e política documentada de atualização de histórias anteriores.
- [ ] Registrar modelo LLM, revisão/checkpoint, prompt, parâmetros, context window, tokenizer, temperatura, hash da entrada/saída e latência por chamada.
- [ ] Executar inicialmente LLM local (Ollama ou outro endpoint compatível), sem fine-tuning, sobre um subconjunto; congelar o modelo antes do benchmark principal.
- [ ] Congelar scoring e limiares escolhidos apenas no validation set. Não mexer no test set para melhorar resultado.
- [ ] Implementar logs e artefatos por execução em `runs/<run_id>/`, preservando as entradas e as evidências recuperadas.
- [ ] Escrever a contribuição como algoritmo/pseudocódigo matemático preciso, e não como só um prompt do LLM.

## 4. Engenharia mínima necessária para experimentar e demonstrar — P1 (algumas partes P0)

- [ ] Empacotar `src/ddpf/detection/` e `src/ddpf/prediction/`, migrando apenas o código legado que é usado por experimentos novos; manter notebooks históricos imutáveis.
- [ ] Criar pipeline configurável com `config.yaml` e CLI (`ddpf describe`, `ddpf detect`, `ddpf predict`, `ddpf stories`, `ddpf evaluate`).
- [ ] Adicionar acessos paginados ao Neo4j, índices e cursores; não materializar grafo nem corpora inteiros em memória.
- [ ] Implementar leitor ZIP/diretório *streaming*, um documento por vez ou mini-lote delimitado, com checksum e deduplicação.
- [ ] Integrar GATE/ANNIE **headless** com inicialização do worker Java pelo Python e verificação de ambiente; `GateWorker(start=True)` depende de instalação/configuração correta, não é instalação automática do GATE.
- [ ] Criar ciclo `read -> annotate -> extract -> persist -> release`, liberando recursos e evitando acumular `RelationMatrix` em dicionário global.
- [ ] Persistir extrações e provenance individuais em SQLite/Parquet, com ID, offsets e checkpoint; mover agregações para etapa sob demanda.
- [ ] Criar resume/retry e isolamento por documento defeituoso sem perder o restante da execução.
- [ ] Separar importação/restauração de ontology/KG da execução habitual: consulta **read-only** e incrementos opcionais, não reconstrução total obrigatória.
- [ ] Criar testes unitários de parser, ontologia, regras, GATE mock, links, árvores, serialização e persistência.
- [ ] Criar teste de integração pequeno `fixtures/` que roda sem o snapshot gigante e teste opcional com Neo4j real.
- [ ] Capturar time, memória máxima RSS, tamanho dos batches, erros recuperados e uso de CPU/GPU.
- [ ] Criar ambiente reprodutível: versões pinadas, lock file, Docker Compose para Neo4j/API/GATE (se viável), comandos CPU fallback e instruções de licenciamento UMLS/GATE/modelos.

## 5. Protocolos de experimento, ablações e análises — P0

- [ ] Pré-registrar configurações e tabela de métodos.
- [ ] M0: baseline simples de similaridade temporal/textual sem KG e sem LLM.
- [ ] M1: grafo/regras sem LLM (knowledge-only).
- [ ] M2: LLM com apenas texto de evento, sem KG e sem regras (LLM-only).
- [ ] M3: LLM + recuperação KG, sem validação de restrições.
- [ ] M4: **método completo** LLM + KG + ontologia + atualizador incremental.
- [ ] Se executável e justo, adicionar Story Forest/EventX original ou variante claramente identificada e documentada; não declarar uma reprodução exata se só usou aproximação.
- [ ] Realizar ablações `-ontology`, `-KG`, `-temporal`, `-spatial`, `-provenance` (somente quando atributo estiver disponível).
- [ ] Usar candidato a vínculo fixo entre métodos na primeira avaliação para isolar qualidade do classificador; depois medir pipeline end-to-end com recuperação de candidatos.
- [ ] Testar pelo menos 3–5 seeds/amostras de execução quando houver aleatoriedade; para LLM, repetir respostas com configuração fixada e reportar variabilidade.
- [ ] Registrar por método precision/recall/F1 por tipo de relação em **rótulos verificáveis**; reportar `unknown`/abstenção e coverage; usar métricas de *partial-label evaluation* quando ground truth não for exaustivo.
- [ ] Medir constraint violation rate, valid-link retention, grounded citation precision e unsupported link rate.
- [ ] Medir relações corretas em histórias e consistência de caminhos; comparar árvores por arestas de referência apenas onde existirem.
- [ ] Medir evolução cronológica sem olhar futuro: por lotes definidos pela data de publicação, não pela data histórica do evento quando é publicada depois.
- [ ] Comparar execução incremental vs reconstrução batch usando mesmas entradas; medir qualidade, diferenças de arestas, latência por lote, tempo total e pico de RAM.
- [ ] Medir custo de inferência (tokens, chamadas, tempo, equipamento), throughput por evento e escala conforme tamanho do conjunto.
- [ ] Separar métricas de tarefas distintas: clustering por doença antigo não equivale a event linking nem story reconstruction.
- [ ] Intervalos de confiança por bootstrap sobre **histórias/fontes independentes** e discussão de significância; não prometer diferenças significativas sem medir.
- [ ] Fazer análise de erros: relações inventadas, temporalmente incompatíveis, duplicadas, sem origem, falsos merges e novos eventos erroneamente inseridos.
- [ ] Verificar resultados em cenário de *label leakage*: remover doença, fonte, ano e referências WHO do input quando atuam como atalhos para a resposta.
- [ ] Gerar tabelas/figuras apenas por scripts versionados (`scripts/make_tables.py`, `scripts/make_figures.py`), nunca copiar números manualmente de logs.

## 6. DDPF Studio: protótipo demonstrável, não validação com usuários — P1

- [ ] Substituir `interface/data.js` ilustrativo por client/API adapter; preservar botão **Demo** bem identificado.
- [ ] Criar backend (`GET /observations`, `/events`, `/stories`, `/stories/{id}`, `/evidence/{id}`, `/runs/{id}`, `POST /stories/build`, `POST /stories/update`).
- [ ] Usar dados reais e read-only do BioPropaPhenKG para as telas Description e Results, com paginação.
- [ ] Em Detection, oferecer upload de ZIP e execução headless do GATE com progresso e estado por documento; implementar ou marcar claramente como indisponível, nunca simular execução real.
- [ ] Em Prediction, incluir seletor de modelo, uso de ontologia, modo batch/incremental, threshold, estimativa de custo e parâmetros salvos.
- [ ] Em Results, adicionar árvores de histórias exploráveis e a posição temporal/geográfica de cada evento.
- [ ] Ao clicar numa aresta, mostrar tipo, confidence, restrições verificadas, IDs/trechos realmente existentes e decisões do validador.
- [ ] Mostrar as relações `unknown`/abstained, não disfarçar incerteza com ligações falsas.
- [ ] Permitir exportar JSON de árvores, proveniência e configuração; exibir timestamp e versão do snapshot.
- [ ] Adicionar visualização da diferença entre árvore gerada e referência para exemplares selecionados.
- [ ] Fazer smoke test visual e de API com 1 história demonstrável real; não afirmar teste de usabilidade sem participantes.
- [ ] Capturar 2–3 figuras reais do sistema para o artigo, sempre identificando se são exemplos ilustrativos ou saídas experimentais.

## 7. Escrever o artigo e o pacote de reprodução — P0

- [ ] Escolher foco **Methodology** (método novo e demonstrável) ou **Research** (resultados científicos novos e amplamente avaliados) de acordo com a evidência produzida.
- [ ] Título de trabalho: **Ontology-Guided LLM Event Linking and Incremental Story Construction for Retrospective Epidemic Surveillance**.
- [ ] Abstract com contexto, lacuna, método, datasets, baselines, métricas e números **reais**; não copiar F1 antigo como validação de história.
- [ ] Introdução: problema + lacuna Story Forest original vs domínio epidemiológico + contribuições verificáveis.
- [ ] Related Work: EBS; ontologias/graphs em vigilância; Story Forest; LLM para event linking; reasoning/grounding e avaliação automatizada.
- [ ] Methods: visão do DDPF anterior, PropaPhen/BioPropaPhenKG, definição formal, recuperação, LLM, validação, formação/atualização de histórias e complexidade.
- [ ] Data & Setup: snapshots Zenodo, corpus, limitações de provenance, critérios dos casos, hardware/modelos/prompts, benchmark silver e splits.
- [ ] Results: tabela comparativa, ablações, resultados de consistência, grounding, incrementalidade, custos e erros; incluir intervalos quando válidos.
- [ ] Discussion: por que a ontologia ajuda ou não; riscos de rótulos temporais, incompletude do gold, dependência de WHO, validade externa e diferença entre narrativa e causalidade.
- [ ] Interface/System: diagrama da arquitetura e 1–2 figuras reais do Studio; não transformar a interface no principal resultado se não houve estudo com usuários.
- [ ] Limitations: ausência de avaliadores humanos, silver labels incompletos, amostra retrospectiva, sem predição clínica/real-time, LLM não garante verdade epidemiológica.
- [ ] Conclusion com achados efetivamente demonstrados e aplicações futuras.
- [ ] Diferenciar claramente achados herdados da tese/artigos anteriores dos **resultados inéditos**; citar artigos, tese e DOIs, evitando republicação substancial e reutilização inadequada de figuras.
- [ ] Criar `data_availability.md`, `code_availability.md`, licença/limitações de fontes UMLS, README de reprodução, prompts, configurações, manifests, scripts e resultados agregados.
- [ ] Congelar release de código com hash/commit e depositar versão no Zenodo com DOI; GitHub sozinho não fornece DOI permanente.
- [ ] Incluir declarações editoriais aplicáveis: funding (obrigatório), competing interests, author contributions, ética/consentimento conforme tipos de dados, agradecimentos e eventual declaração de uso de IA.
- [ ] Verificar permissões de uso e reprodução de dados, imagens de terceiros e referências do Story Forest (não reproduzir figuras sem licença/permissão).
- [ ] Conferir requisitos do periódico no dia de submissão: abstract <250 palavras, cover letter, figura/tabelas, política editorial e APC/financiamento.
- [ ] Revisar inglês, nomenclatura, tabelas e referências; remover linguagem não sustentada ('causal', 'real-time', 'early warning', 'diagnosis').
- [ ] Garantir que outro pesquisador consiga reproduzir **pelo menos um subconjunto** apenas com README, snapshot acessível e comandos documentados.
- [ ] Escrever cover letter mostrando **qual é a nova contribuição em relação à tese e ao manuscrito rejeitado**; explicar por que não é só aplicar clustering.
- [ ] Validar consentimento/autoria e submeter; guardar submissão e arquivos finais.

## 8. Trabalhos opcionais para versão seguinte — P2

- [ ] Fine-tuning de LLM de domínio (apenas se baselines zero-shot forem insuficientes e houver dados adequados).
- [ ] Engenharia completa do modo live/stream online (além de replay temporal retrospectivo).
- [ ] Avaliação humana de utilidade/usabilidade e fluxo de vigilância aplicada (para um futuro artigo orientado à IJMI).
- [ ] Terceira doença e novos países/períodos, se existirem dados comparáveis com referências independentes.
- [ ] Comparar Story Tree vs DAG e múltiplas árvores/histórias com métricas adicionais.
- [ ] Melhorias visuais e autenticação/controle de acesso para implantação externa.

## Ordem e decisões (plano de execução solo, ilustrativo)

1. **Semana 1: Gate A** — restaurar snapshot, auditoria de atributos e exportar 30–50 eventos com provenance. Se não houver individualização, recuperar pequeno conjunto de documentos ou mudar escopo.
2. **Semana 2: Gate B** — escrever benchmark silver com referência independente e negativo sintético separado; pré-registrar splits e baselines.
3. **Semanas 3–4:** método mínimo, LLM local, ontologia/rules, 4–5 variantes, primeira tabela de métricas; **Gate C** de contribuição real.
4. **Semanas 5–6:** incrementalidade, ablações, custo, validação temporal, análise de erros e consolidação reprodutível.
5. **Semanas 7–8:** integrar resultados reais no Studio, escrever e diagramar artigo, statements e submissão.

Os prazos são apenas uma estimativa de organização. Podem aumentar significativamente se o snapshot não permitir recuperar evidência individual, o benchmark exigir obtenção de fontes ou as baselines não forem reprodutíveis.

## Checklist curto: condições para dizer 'pronto para submeter'

- [ ] Resultado novo não é apenas mais uma aplicação de clustering.
- [ ] O benchmark tem rótulos **não circulares** e separa `unknown` de falsos.
- [ ] A ontologia é efetivamente utilizada pelo método e testada por ablação.
- [ ] Há pelo menos uma comparação LLM-only, uma sem LLM e a configuração completa.
- [ ] As árvores podem ser rastreadas até evidência registrada; o texto não atribui causalidade indevida.
- [ ] Resultados de qualidade e custo estão completos, com análise de erros/limitações.
- [ ] Dados e código possuem links, versão, instruções e condições de acesso claras.
- [ ] O Studio apresenta saídas reais ou declara claramente seu modo demonstrativo.
- [ ] Artigo e cover letter demonstram diferença verificável em relação às versões anteriores.
