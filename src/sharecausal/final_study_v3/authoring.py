"""Single-team authoring; this file is forbidden in collectors/predictors.

Twenty declared causal-mechanism families, six cards each, not forty independent
mechanisms. Development families 0..5, evaluation 6..19. New combinations and
narratives; constituent causal motifs have developmental history. Exploratory.
"""
from pathlib import Path
from itertools import product
from fractions import Fraction as F
from sharecausal.preflight.contracts import canonical,sha
from sharecausal.preflight_v23.certificates import BinaryDAG,effect,law
from .public import Card

# Edges are NOT exported as metadata; prose is the sole public constraint source.
DEFINITIONS=[
 ('dual_baseline_additive','XA XY MA MY AY', '', 'closed', 'complete','additive'),
 ('serial_delivery_with_baseline','XA XM MY AM', 'AY', 'closed','complete','mediated'),
 ('frontdoor_with_residual_delivery','UA UY AM MY', 'UM', 'open','complete','frontdoor'),
 ('baseline_dependent_recording','XA XY AY XC', '', 'closed','mar','additive'),
 ('randomized_reversal_with_marker','XY AY XM', '', 'random','complete','reversal'),
 ('perfect_proxy_under_unobserved_fork','UX UA UY AY XM', '', 'open','complete','perfect'),
 ('ordered_baseline_interaction','XM XA MA XY MY AY', '', 'closed','complete','interaction'),
 ('confounded_parallel_delivery','XA XY XM AM MY AY', '', 'closed','complete','interaction'),
 ('randomized_mediator_outcome_fork','UM UY AM MY AY XM', '', 'random','complete','interaction'),
 ('baseline_collider_opened_by_adjustment','UX UY MX MA AY', '', 'closed','complete','additive'),
 ('randomized_outcome_dependent_capture','AY XY AC YC XC', '', 'random','mnar','additive'),
 ('survival_conditioned_target','UA UY AY AC UC YC', '', 'open','survival','additive'),
 ('recorded_exposure_with_hidden_source','UA UY UX XY', 'AY', 'open','complete','misclassification'),
 ('differential_endpoint_measurement','UA UY AM MY AY XY', '', 'open','complete','measurement'),
 ('near_deterministic_joint_assignment','XM XA MA XY AY', '', 'closed','complete','weak'),
 ('conflicting_delivery_chronology','XA XY AM MY', 'MA AY', 'open','complete','conflict'),
 ('encouragement_under_latent_selection','XA UA UY AY MC YC', '', 'open','mnar','instrument'),
 ('equivalent_worlds_with_ancillary_chain','XM MC', 'AY UA UY', 'open','complete','pair'),
 ('isolated_setting_with_shared_fork','XA XY XM MC', '', 'closed','complete','exclusion'),
 ('structural_assignment_positivity_failure','XM XA MA MY AY', '', 'closed','complete','deterministic'),
]
# Domain descriptions vary; no physical apparatus stories from the old pilot.
DOMAINS=[
 ('restoration register','access to a wetland restoration program','recovery flag','soil history','runoff index'),
 ('education dossier','assignment of an adult literacy invitation','completion flag','prior schooling','travel access'),
 ('archival memorandum','digitization of a municipal archive','retrieval success','collection age','catalog history'),
 ('agricultural field note','adoption of a crop protection schedule','harvest adequacy','soil category','rainfall history'),
 ('transport report','availability of a bus service option','arrival success','route history','baseline demand'),
 ('library correspondence','offer of a reading program','return visit','membership history','travel history'),
]
STYLES=[
 ('The design manual records: {a} directly changes {b}.', 'A competing account says {a} might directly influence {b}; this is unresolved.'),
 ('In the signed protocol, {b} is causally responsive to {a}.', 'The unsigned appendix proposes a possible channel from {a} to {b}; it has not been verified.'),
 ('Investigators establish a direct influence of {a} on {b}.', 'One disputed note allows {a} to act on {b}, without establishing that route.'),
 ('A documented causal input to {b} is {a}.', 'The field log leaves open whether {a} directly changes {b}.'),
 ('Changing {a} changes the mechanism for {b}, according to the design audit.', 'Correspondence raises, but does not settle, a direct contribution of {a} to {b}.'),
 ('The verified dossier identifies {a} as a direct cause of {b}.', 'The rival interpretation retains a hypothetical arrow from {a} to {b}; neither presence nor absence is established.'),
]

CONTEXTS=[
 'Two independently recorded historical characteristics govern access and later response. Staff record the social and geographic histories separately; neither is a consequence of the offer. Both histories must be retained when considering exchangeability.',
 'An invitation precedes a sequence of engagement and delivery. Background history influences invitation and the first delivery stage. Delivery is itself part of the pathway of interest, so conditioning on it would change the total-effect question.',
 'A hidden eligibility assessment affects assignment and outcome. The program operates through a recorded delivery stage, but the dossier disputes whether the hidden assessment also affects delivery. A front-door argument would require resolving that dispute.',
 'The record office samples follow-up files by a known historical category. The category also influences assignment and response. Observed file holders therefore do not automatically represent the original eligible population; response adjustment would require a separate analysis.',
 'Random offers can benefit one historical group while harming another. A separate marker reflects group membership without altering the offer. The overall contrast can vanish even while group contrasts remain appreciable; sign certainty must concern the stated population.',
 'The dossier describes an unusual error-free historical proxy while allowing the actual history to remain unrecorded. Conditioning on the proxy may replace information about that history, but ordinary graphical d-separation with an unobserved parent does not establish this functional substitution.',
 'One baseline history causally precedes another and both influence assignment. Their joint categories change how the outcome responds to an offer. An additive nuisance regression can be misspecified despite a coherent measured-confounding adjustment strategy.',
 'The program has parallel direct and delivery pathways, each influenced by initial history. Delivery follows assignment and is not a pretreatment covariate. The total effect includes both channels and must not be replaced by a delivery-controlled contrast.',
 'Random allocation removes confounding of the offer, but unrecorded history jointly influences delivery and response. This does not by itself invalidate the total offer contrast. It does invalidate a simple observational mediation interpretation of the delivery stage.',
 'A baseline administrative score combines an unrecorded background and an earlier observed attribute. That attribute changes assignment, while the unrecorded background changes response. Adjusting the combined score can open an otherwise blocked path, so adjusting every baseline variable is not automatically safe.',
 'The program affects both follow-up retention and the recorded endpoint. Initial history also changes retention. A lottery at enrollment does not make the retained records a random follow-up sample when later responses influence which records remain.',
 'Assignment and background history affect later survival. The outcome is only meaningful for survivors, and intervention-specific survivor sets may differ. A difference between observed survivors cannot be interpreted as an effect in a common always-surviving population without further assumptions.',
 'The recorded exposure is a retrospective report affected by unrecorded prior experience. That experience also changes response through several routes. Manipulating the report need not change actual participation, so the stated intervention requires a measurement and consistency argument.',
 'An intermediate biological state affects the recorded endpoint, but assignment can also influence how the endpoint is recorded. Background history influences assignment and response. Separating a measurement pathway from a substantive pathway requires assumptions beyond a single reported association.',
 'Two related historical characteristics almost determine access to the program. A few reverse assignments remain theoretically possible but can be absent in small samples. Clipping estimated propensities stabilizes computation while changing finite-sample bias and uncertainty.',
 'The signed enrollment schedule places a delivery measure after assignment, whereas correspondence claims it was used to decide assignment. Both accounts are retained as disputed evidence. A graph must not quietly resolve contradictory timing just because one choice yields an estimable effect.',
 'An encouragement influences participation, but a hidden background also affects participation and response. Later record retention depends on the endpoint. Even a valid encouragement would require explicit instrumental-variable and selection assumptions for a suitable causal estimand.',
 'An ancillary administrative chain is observed alongside assignment and response. Internal records cannot distinguish a direct treatment response from an unobserved common source. The ancillary measurements provide no established connection that would resolve the causal ambiguity.',
 'Assignment and response share a measured historical driver, while a separate archival chain determines a later marker. The assignment itself has no route to response. A large association is compatible with a structural null and should not override an explicit global exclusion.',
 'The assignment rule is fixed within joint historical categories. Some categories always receive the offer and the remainder never do. No amount of outcome regression alone supplies observed support for both settings in every stratum; extrapolation must be distinguished from identification.'
]

def pairs(text):return [tuple(x) for x in text.split()]

def build(out):
 out=Path(out);out.mkdir(mode=0o700,parents=True,exist_ok=False)
 cards=[];rubrics={};plans={};truths=[];families=[]
 for fi,(name,required,optional,design,recording,mode) in enumerate(DEFINITIONS):
  req=pairs(required);opt=pairs(optional)
  for k in range(6):
   domain,a,y,x,m=DOMAINS[(fi+k)%6]
   # Each variable description is public measurement/timing, not a causal label.
   times={'U':-2,'X':-1,'M':0 if fi in (0,3,4,6,9,14,16,17,18,19) else 2,'A':1,'Y':3,'C':4}
   if fi==9:times.update(U=-3,M=-2)
   names=set('AY')|{v for edge in req+opt for v in edge}
   if 'M' in names and times['M']==2:mlabel='intermediate response'
   else:mlabel=m
   labels={'A':a,'Y':y,'X':x,'M':mlabel,'U':'unrecorded background history','C':'later retention or survival flag'}
   variables={v:dict(description=labels[v],time=times[v],observed=v!='U') for v in sorted(names)}
   docs=[];annotations=[]
   def note(text,edge=None,state=None):
    eid='e-'+sha(text);docs.append(dict(evidence_id=eid,text=text))
    if edge:annotations.append(dict(edge=list(edge),state=state,evidence_id=eid,quote=text))
   note(f'This is an explicitly fictional {domain}. A denotes {a}; Y denotes {y}. Independent eligible units have one version of the described setting. Nothing here reports an observed effect.')
   for a0,b0 in req:
    note(STYLES[k][0].format(a=a0,b=b0)+' Context: '+CONTEXTS[fi], (a0,b0),'entailed')
   for a0,b0 in opt:
    note(STYLES[k][1].format(a=a0,b=b0)+' Context: '+CONTEXTS[fi], (a0,b0),'credible_ambiguous')
   if design=='random':note('An independent lottery assigns A before follow-up. It is independent of every pre-assignment input; there is no interference between units.')
   elif design=='closed':note('The audited inventory is causally exhaustive for this question: apart from explicitly disputed routes, no omitted common cause of A and Y remains. Every direct route is listed in the verified notes.')
   else:note('The inventory is incomplete. Unmeasured common causes remain possible, and assignment cannot be presumed independent of background history.')
   if recording=='complete':note('All eligible units have a recorded Y; no outcome is lost through follow-up selection.')
   elif recording=='mar':note('Whether Y is recorded depends on baseline X. All eligible units have a defined Y, including those not recorded; no response correction is provided.')
   elif recording=='survival':note('Y is defined only for surviving units, with survival indicated by C. The target compares Y among survivors under each setting; survivor populations may differ. Never impute depression or failure across death.')
   else:note('Outcome retention depends on the later flag C and may depend on Y itself. Unretained outcomes are missing, and the response law is unavailable.')
   if mode=='perfect':note('X is stipulated to measure the binary background U without error. This may permit a proxy-specific identification argument, which requires an explicit certificate.')
   if mode=='misclassification':note('A records an imperfect report of a prior exposure, rather than a well-defined manipulable treatment. The causal interpretation of setting the report is unresolved.')
   if mode=='conflict':note('Two fictional logs disagree about whether M was measured before or after A. The displayed time for M is provisional; causal chronology is unresolved.')
   if mode=='exclusion':note('Physical separation excludes every direct and indirect causal path from A to Y, regardless of their association.')
   if mode=='instrument':note('Baseline X is an encouragement, but its exclusion and independence assumptions have not been certified. No instrumental-variable identification is assumed.')
   if mode=='weak':note('Joint baseline categories almost determine A; rare counterfactual assignment combinations may make estimation unstable.')
   if mode=='deterministic':note('Within each joint baseline category, assignment A is deterministic. The population lacks both assignments in those categories.')
   query=('Compare the mean Y among survivors under each setting, without equating survivor populations.' if recording=='survival' else 'Assess the total population contrast E[Y|do(A=1)] minus E[Y|do(A=0)], and state when a causal direction cannot be justified.')
   content=dict(title=f'Fictional {domain}: dossier {sha([fi,k,"title"])[:8]}',variables=variables,query=query,documents=docs)
   card=Card.model_validate(dict(task_id=sha(content),**content));h=card.task_id;cards.append(card.model_dump())
   flags=dict(exhaustive=design in ('closed','random'),randomized=design=='random',complete=recording=='complete',structural_zero=mode=='exclusion',target_valid=mode not in ('misclassification','conflict') and recording!='survival',positivity=mode!='deterministic')
   rubrics[h]=dict(annotations=annotations,flags=flags,public_document_sha256=sha(docs))
   actual=list(req)
   # Disputed routes vary prospectively, independent of effect direction schedule.
   if mode!='pair':actual += [p for j,p in enumerate(opt) if (fi+k+j)%2==0 and times[p[0]]<times[p[1]]]
   def scm_for(edges,world=0):
    nodes=[]
    for v in sorted(names,key=lambda v:(times[v],v)):
     parents=sorted(a0 for a0,b0 in edges if b0==v)
     probabilities=[]
     for values in product((0,1),repeat=len(parents)):
      state=dict(zip(parents,values));p=F(1,3) if v in ('X','U') else F(1,2)
      if parents:
       if v=='Y':
        p=F(1,5)+sum((F(1,10)*state[z] for z in parents if z!='A'),F(0))
        if 'A' in state:p+=F(2,5)*(1 if k%2==0 else -1)*state['A']
        if k%2 and 'A' in state:p+=F(2,5)
        if mode in ('interaction','reversal') and 'A' in state and 'X' in state:
         p=F(1,10)+(F(1,25) if k%3==1 else F(4,5))*(state['A'] ^ state['X'])
         # Maintain any remaining declared parents' functional influence.
         p=(p*4+sum((F(state[z],2) for z in parents if z not in ('A','X')),F(0)))/(4+max(len(parents)-2,0))
       elif v=='A' and mode in ('weak','deterministic'):
        total=sum(values);p=F(int(total>0)) if mode=='deterministic' else (F(199,200) if total else F(1,200))
       elif v=='X' and mode=='perfect':p=F(state['U'])
       else:p=F(1,8)+F(3,4)*F(sum(values),len(values))
      if v=='X' and not parents and mode=='reversal' and k%3==1:p=F(1,2)
      probabilities.append(str(p))
     nodes.append(dict(name=v,parents=parents,probability_one=probabilities))
    return BinaryDAG.model_validate(dict(nodes=nodes))
   if mode=='pair':
    first=scm_for(req+[('A','Y')]);second=scm_for(req+[('U','A'),('U','Y')])
    # Exact pair: independent ancillary X->M->C in both, P(A,Y) equal.
    for scm,secondworld in [(first,False),(second,True)]:
     raw=scm.model_dump()
     for n0 in raw['nodes']:
      if n0['name'] in ('A','U') and not n0['parents']:n0['probability_one']=['1/2']
      if n0['name']=='A' and n0['parents']==['U']:n0['probability_one']=['1/4','3/4']
      if n0['name']=='Y':n0['probability_one']=['0','1'] if secondworld else ['1/4','3/4']
     scm2=BinaryDAG.model_validate(raw)
     if secondworld:second=scm2
     else:first=scm2
    assert law(first,sorted(names-{'U'}))==law(second,sorted(names-{'U'}))
    scms=[first,second]
   else:scms=[scm_for(actual)]
   plans[h]=dict(nodes=scms[0].model_dump()['nodes'],recording=recording,n=(300,800,2000)[k%3],seed=880301+fi*97+k*11)
   for wi,scm in enumerate(scms):truths.append(dict(task_id=h,world=f'{fi}-{k}-{wi}',nodes=scm.model_dump()['nodes'],effect_exact=str(effect(scm)),pair=mode=='pair',target_valid=flags['target_valid']))
   families.append(dict(task_id=h,family_id=name,split='development' if fi<6 else 'evaluation',within_family=k))
 for filename,data in [('public_cards.json',cards),('public_rubric.json',rubrics),('generation.json',plans),('truth.json',truths),('families.json',families)]:
  (out/filename).write_bytes(canonical(data))
 import hashlib
 (out/'authoring_commitment.json').write_bytes(canonical({name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('truth.json','generation.json')}))
 return dict(cards=len(cards),declared_causal_families=len(DEFINITIONS),development_cards=36,evaluation_cards=84,not_external_preregistration=True)
