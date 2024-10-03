from transformers import BitsAndBytesConfig
from torch import bfloat16
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

World_Cup_File = 'Notebook/cricket_summary.csv'
prompts = [
    ['summary',"As a sports journalist give textual summary of above match from data provided above in 1 paragraph"],
    ['bolling_summary',"As a sports journalist give summary of bowler's preformance of both teams in 1 paragraph"],
    ['batting_summary',"As a sports journalist give a summary of batter's preformance of both teams in 1 paragraph"],
    ['team1_support',"As ##team1## supporter give summary of the above match data in one paragraph"],
    ['team2_support',"As @@team2@@ supporter give summary of the above match data in one paragraph"]
]
prompt_template=["{article} , using above information as context give a descriptive summary of the match in one paragraph generate answer only from given article.\n",
                 "{article} , using above information as context act as a sports journalist and give a summary of all the bowler's performance in one paragraph generate answer only from given article.\n",
                 "{article} , using above information as context act as a sports journalist and give a descriptive summary of all the batter's performance in one paragraph generate answer only from given article.\n",
                 "{article} , using above information as context give your views on batting and bowling of player fom the perspective of the first batting teams fan in one paragraph generate answer only from given article.\n",
                 "{article} , using above information as context from the perspective of the second batting teams fan give your views on batting and bowling of their team in one paragraph generate answer only from given article.\n",
                 ]


modelDict = {
    'llama2':'/home/nlp-08/Downloads/ODI-WORLDCUP/model/llama2',
    'llama3':'/home/nlp-08/Downloads/ODI-WORLDCUP/model/llama3',
    'mistral':'/home/nlp-08/Downloads/ODI-WORLDCUP/model/Mistral-7B',
    'Phi-3-mini-128k-instruct':'/home/nlp-08/Downloads/ODI-WORLDCUP/model/Phi-3-mini-128k-instruct',
    'Phi-3-small-128k-instruct':'microsoft/Phi-3-small-128k-instruct',
    'models--Qwen--Qwen1.5-7B-Chat':'/home/nlp-08/Downloads/ODI-WORLDCUP/model/models--Qwen--Qwen1.5-7B-Chat',
    'vicuna-7b-v1.5':'/home/nlp-08/Downloads/ODI-WORLDCUP/model/vicuna-7b-v1.5',
    'tiiuae--falcon-7b-instruct':'/home/nlp-08/Downloads/ODI-WORLDCUP/model/tiiuae--falcon-7b-instruct',
    'tiny llama':'/home/nlp-08/Downloads/ODI-WORLDCUP/model/tiny_llama'

}

prompt_history = '''
Analyse the given match report and JSON data, and use your cricket knowledge to generate a one-paragraph match summary. Ensure only the key facts explicitly mentioned in the data are included, while applying cricket insights on what should be highlighted in a summary, such as important statistics, key performances, turning points, and the match outcome. Keep the summary concise and accurate.
Analyse the match report & json data, based on them generate a cricket match summary in 1 paragraph only,make summary such that important info will be included nothing else ?
Analyse the given match report & JSON data, and use your cricket knowledge to generate a one-paragraph match summary. Ensure only the key facts explicitly mentioned in the data are included, while applying cricket insights on what should be highlighted in a summary, such as important statistics, key performances, turning points, and the match outcome. Keep the summary concise and accurate.
Analyse the given JSON data & generate a one-paragraph match summary. Include key facts, starting with the first innings, followed by the second. Highlight important statistics, performances, turning points, and the match outcome. Keep it concise and accurate.
Analyse the given match report & JSON data, and use cricket knowledge to generate a one-paragraph match summary. Apply cricket insights to highlight important statistics, key performances and the match outcome. Keep the summary concise and accurate.
'''

di={
    "summary":"Match Summary",
    "bolling_summary":"Bowler Summary",
    "batting_summary":"Batter Summary",
    "team1_support":"Team_1 Support",
    "team2_support":"Team_2 Support",
}
def fillScore(name, answer, human_genrated ):
    result = {}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    bert_scorer = BERTScorer(model_type='bert-base-uncased')
    bleu_score = sentence_bleu([human_genrated], answer, smoothing_function=SmoothingFunction().method1)
        
    bert_P, bert_R, bert_F1  = bert_scorer.score([answer], [human_genrated])    # ([answer_text],[j[k[0]]])
    scores = scorer.score(answer, human_genrated)                               # (answer_text, j[k[0]])
    print(di[name]+' Generated')
    result[di[name]+' Generated'] = answer
    result[di[name] + ' rouge l Precision'] = scores['rougeL'].precision
    result[di[name] + ' rouge l Recall'] = scores['rougeL'].recall
    result[di[name] + ' rouge l F1_measure'] = scores['rougeL'].fmeasure
    result[di[name] + ' bert Precision'] = f"{bert_P.mean():.4f}"
    result[di[name] + ' bert Recall'] = f"{bert_R.mean():.4f}"
    result[di[name] + ' bert F1_measure'] = f"{bert_F1.mean():.4f}"
    result[di[name] + ' bleu score'] = bleu_score
    return result

def avg_rouge_l_Precision (df):
    l= df['Match Summary rouge l Precision'].mean()
    m=df['Bowler Summary rouge l Precision'].mean()
    n=df['Batter Summary rouge l Precision'].mean()
    o=df['Team_1 Support rouge l Precision'].mean()
    p=df['Team_2 Support rouge l Precision'].mean()
    return (l+m+n+o+p)/5


def avg_rouge_l_Recall (df):
    l= df['Match Summary rouge l Recall'].mean()
    m=df['Bowler Summary rouge l Recall'].mean()
    n=df['Batter Summary rouge l Recall'].mean()
    o=df['Team_1 Support rouge l Recall'].mean()
    p=df['Team_2 Support rouge l Recall'].mean()
    return (l+m+n+o+p)/5


def avg_rouge_l_Fmeasure (df):
    l= df['Match Summary rouge l F1_measure'].mean()
    m=df['Bowler Summary rouge l F1_measure'].mean()
    n=df['Batter Summary rouge l F1_measure'].mean()
    o=df['Team_1 Support rouge l F1_measure'].mean()
    p=df['Team_2 Support rouge l F1_measure'].mean()
    return (l+m+n+o+p)/5


def avg_bleu_score (df):
    l= df['Match Summary bleu score'].mean()
    m=df['Bowler Summary bleu score'].mean()
    n=df['Batter Summary bleu score'].mean()
    o=df['Team_1 Support bleu score'].mean()
    p=df['Team_2 Support bleu score'].mean()
    return (l+m+n+o+p)/5

def avg_bert_Precision (df):
    l= df['Match Summary bert Precision'].mean()
    m=df['Bowler Summary bert Precision'].mean()
    n=df['Batter Summary bert Precision'].mean()
    o=df['Team_1 Support bert Precision'].mean()
    p=df['Team_2 Support bert Precision'].mean()
    return (l+m+n+o+p)/5

def avg_bert_Recall (df):
    l= df['Match Summary bert Recall'].mean()
    m=df['Bowler Summary bert Recall'].mean()
    n=df['Batter Summary bert Recall'].mean()
    o=df['Team_1 Support bert Recall'].mean()
    p=df['Team_2 Support bert Recall'].mean()
    return (l+m+n+o+p)/5


def avg_bert_Fmeasure (df):
    l= df['Match Summary bert F1_measure'].mean()
    m=df['Bowler Summary bert F1_measure'].mean()
    n=df['Batter Summary bert F1_measure'].mean()
    o=df['Team_1 Support bert F1_measure'].mean()
    p=df['Team_2 Support bert F1_measure'].mean()
    return (l+m+n+o+p)/5


def make_dict (df,name):
    d={}
    d['model']=name
    d['Rouge-l Precision']=avg_rouge_l_Precision(df)
    d['Rouge-l Recall']=avg_rouge_l_Recall(df)
    d['Rouge-l F1-score']=avg_rouge_l_Fmeasure(df)
    d['Bleu Score']=avg_bleu_score(df)
    d['Bert Precision']=avg_bert_Precision(df)
    d['Bert Recall']=avg_bert_Recall(df)
    d['Bert F1-score']=avg_bert_Fmeasure(df)
    return d