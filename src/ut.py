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

match = '''
ndia vs Australia, Final - Cricket Score
match_result,Australia won by 6 wkts
team,IND 
score,240-10
overs,50
player,dismissal,runs,balls,fours,sixes,strike_rate
Rohit (c),"c Head b Maxwell",47,31,4,3,151.61
Shubman Gill,"c Zampa b Starc",4,7,0,0,57.14
Kohli,"b Cummins",54,63,4,0,85.71
Shreyas Iyer,"c Josh Inglis b Cummins",4,3,1,0,133.33
Rahul (wk),"c Josh Inglis b Starc",66,107,1,0,61.68
Ravindra Jadeja,"c Josh Inglis b Hazlewood",9,22,0,0,40.91
Suryakumar Yadav,"c Josh Inglis b Hazlewood",18,28,1,0,64.29
Shami,"c Josh Inglis b Starc",6,10,1,0,60
Bumrah,"lbw b Zampa",1,3,0,0,33.33
Kuldeep Yadav,"run out (Labuschagne/Cummins)",10,18,0,0,55.56
Siraj,not out,9,8,1,0,112.5
extras,byes,leg_byes,wides,no_balls,penalty
,,3,9,0,0
total,240-10
overs_played,50
run_rate,4.8
bowler,O,M,R,W,NB,WD,ECO
Starc,10,0,55,3,0,4,5.5
Hazlewood,10,0,60,2,0,1,6
Maxwell,6,0,35,1,0,0,5.8
Cummins (c),10,0,34,2,0,2,3.4
Zampa,10,0,44,1,0,2,4.4
Mitchell Marsh,2,0,5,0,0,0,2.5
Head,2,0,4,0,0,0,2
fall_of_wickets,score,over,batsman
Shubman Gill,30-1,4.2
Rohit,76-2,9.4
Shreyas Iyer,81-3,10.2
Kohli,148-4,28.3
Ravindra Jadeja,178-5,35.5
Rahul,203-6,41.3
Shami,211-7,43.4
Bumrah,214-8,44.5
Suryakumar Yadav,226-9,47.3
Kuldeep Yadav,240-10,50
powerplays,overs,runs,mandatory
0.1 - 10,80,AUS
score,241-4
overs,43
player,dismissal,runs,balls,fours,sixes,strike_rate
David Warner,"c Kohli b Shami",7,3,1,0,233.33
Travis Head,"c Shubman Gill b Siraj",137,120,15,4,114.17
Mitchell Marsh,"c Rahul b Bumrah",15,15,1,1,100
Steven Smith,"lbw b Bumrah",4,9,1,0,44.44
Marnus Labuschagne,not out,58,110,4,0,52.73
Glenn Maxwell,not out,2,1,0,0,200
extras,byes,leg_byes,wides,no_balls,penalty
5,2,11,0,0
total,241-4
overs_played,43
run_rate,5.6
bowler,O,M,R,W,NB,WD,ECO
Jasprit Bumrah,9,2,43,2,0,0,4.8
Mohammed Shami,7,1,47,1,0,9,6.7
Ravindra Jadeja,10,0,43,0,0,2,4.3
Kuldeep Yadav,10,0,56,0,0,0,5.6
Mohammed Siraj,7,0,45,1,0,0,6.4
fall_of_wickets,score,over,batsman
David Warner,16-1,1.1
Mitchell Marsh,41-2,4.3
Steven Smith,47-3,6.6
Travis Head,239-4,42.5
powerplays,overs,runs,mandatory
0.1 - 10,60
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