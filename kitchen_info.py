# stores notes and numbers about the kitchen environment
# for reward function: https://github.com/rail-berkeley/d4rl/blob/master/d4rl/kitchen/kitchen_envs.py

import numpy as np

task_dict = {'m':'microwave', 'k':'kettle', 'b':'bottom burner', 'c':'slide cabinet'}

# Below are qpos for the gripper for interacting with the task object (for expressing preference), the first 9 number from the state
BEGIN_MICROWAVE = np.array([-1.3327, -1.7641,  1.9147, -1.2338, -0.4308,  1.0447,  2.2243,  0.0310, 0.0259])
FINISH_MICROWAVE = np.array([-0.7265, -1.7677,  1.9274, -1.4496, -0.0352,  0.9504,  1.9905,  0.0146, 0.0260]) 
MICROWAVE_WAYPOINTS = np.array([[-1.3514217138290405, -1.7641912698745728, 1.9187260866165161, -1.2217079401016235, -0.4589982032775879, 1.019494891166687, 2.2157888412475586, 0.02036968246102333, 0.0142771415412426] ,
                                [-1.332749605178833, -1.7641199827194214, 1.914694905281067, -1.2337548732757568, -0.43080902099609375, 1.044731616973877, 2.2243266105651855, 0.030992940068244934, 0.025888698175549507] ,
                                [-1.3253737688064575, -1.7623039484024048, 1.9165596961975098, -1.232474684715271, -0.41662076115608215, 1.037234902381897, 2.2260074615478516, 0.03419444337487221, 0.016943613067269325] ,
                                [-1.2949835062026978, -1.763317346572876, 1.9092411994934082, -1.2457990646362305, -0.3842947781085968, 1.0393604040145874, 2.2218000888824463, 0.04514143243432045, 0.010683691129088402] ,
                                [-1.2665866613388062, -1.760008454322815, 1.9217958450317383, -1.2750859260559082, -0.356300950050354, 1.0315568447113037, 2.1915247440338135, 0.04722002521157265, 0.008345982991158962] ,
                                [-1.2253036499023438, -1.760615348815918, 1.9182264804840088, -1.291345477104187, -0.31575173139572144, 1.0180917978286743, 2.1684160232543945, 0.03282018005847931, 0.002848014933988452] ,
                                [-1.1932202577590942, -1.7713711261749268, 1.9243178367614746, -1.3041331768035889, -0.28737446665763855, 1.0081136226654053, 2.1576485633850098, 0.030981244519352913, -0.001010563806630671] ,
                                [-1.1633764505386353, -1.7715504169464111, 1.910749077796936, -1.3147386312484741, -0.25645390152931213, 0.9661275744438171, 2.1305487155914307, 0.044140275567770004, 0.0067167519591748714] ,
                                [-1.1333966255187988, -1.7611583471298218, 1.9118467569351196, -1.3222090005874634, -0.20678606629371643, 0.9287218451499939, 2.1083860397338867, 0.04093371704220772, 0.005969277583062649] ,
                                [-1.1015640497207642, -1.7676531076431274, 1.9264901876449585, -1.3289544582366943, -0.16855721175670624, 0.8954817056655884, 2.07477068901062, 0.04839020222425461, -0.005755910184234381] ,
                                [-1.0737215280532837, -1.7726949453353882, 1.9258326292037964, -1.3243106603622437, -0.13293258845806122, 0.8569347262382507, 2.0592010021209717, 0.04864592105150223, 0.010830657556653023] ,
                                [-1.065237283706665, -1.7621958255767822, 1.9322571754455566, -1.323501706123352, -0.11617869138717651, 0.8420128226280212, 2.053227424621582, 0.03572206199169159, 0.007189149037003517] ,
                                [-1.0536003112792969, -1.7575269937515259, 1.9300141334533691, -1.3215360641479492, -0.1199149340391159, 0.8313848972320557, 2.063066005706787, 0.04170963168144226, 0.009191885590553284] ,
                                [-1.0378952026367188, -1.7655333280563354, 1.9155932664871216, -1.320543885231018, -0.11121925711631775, 0.8168821930885315, 2.056945562362671, 0.04804769903421402, 0.00557415746152401] ,
                                [-1.038317084312439, -1.7581188678741455, 1.9231541156768799, -1.3184266090393066, -0.08893750607967377, 0.804839015007019, 2.065608263015747, 0.0435602068901062, 0.008956784382462502] ,
                                [-1.0366036891937256, -1.7630048990249634, 1.9143364429473877, -1.310973048210144, -0.0873015820980072, 0.7904003262519836, 2.0731143951416016, 0.03845503181219101, 0.005543524399399757] ,
                                [-1.0277056694030762, -1.7563079595565796, 1.9085773229599, -1.313873529434204, -0.07321066409349442, 0.768822431564331, 2.076897621154785, 0.039293840527534485, 0.007629863452166319] ,
                                [-1.0129226446151733, -1.7734581232070923, 1.9029179811477661, -1.3109135627746582, -0.06693034619092941, 0.7717835307121277, 2.0793392658233643, 0.04639656841754913, 0.00022740502026863396] ,
                                [-1.0100677013397217, -1.7737218141555786, 1.8976231813430786, -1.3059240579605103, -0.048121679574251175, 0.7614292502403259, 2.089254856109619, 0.03484461084008217, -0.0013375642010942101] ,
                                [-0.9943733811378479, -1.7647382020950317, 1.8902596235275269, -1.3008341789245605, -0.04581032320857048, 0.7496898174285889, 2.0976295471191406, 0.035810623317956924, 0.01061208639293909] ,
                                [-0.9788323640823364, -1.7672168016433716, 1.8849128484725952, -1.3096598386764526, -0.03118257038295269, 0.7472068071365356, 2.0906195640563965, 0.0381246954202652, 0.006884872913360596] ,
                                [-0.9628305435180664, -1.7636762857437134, 1.899068832397461, -1.3270671367645264, -0.0319860465824604, 0.7745046019554138, 2.0921170711517334, 0.03520220145583153, -0.0008872209000401199] ,
                                [-0.927205502986908, -1.764190673828125, 1.8926182985305786, -1.3492087125778198, -0.027054931968450546, 0.7968026399612427, 2.0873863697052, 0.0383896641433239, 0.01338168140500784] ,
                                [-0.9036985635757446, -1.7721658945083618, 1.8981107473373413, -1.3644384145736694, -0.03546639531850815, 0.8272960782051086, 2.0704939365386963, 0.033562492579221725, 0.007989597506821156] ,
                                [-0.8804124593734741, -1.7665271759033203, 1.9105876684188843, -1.3866446018218994, -0.04904773458838463, 0.8403662443161011, 2.074917793273926, 0.04450725018978119, 0.010796552523970604] ,
                                [-0.849609375, -1.7696210145950317, 1.9094537496566772, -1.394024133682251, -0.05351458862423897, 0.8755301237106323, 2.0643579959869385, 0.030918799340724945, 0.0025715057272464037] ,
                                [-0.8390979170799255, -1.7665427923202515, 1.920856237411499, -1.3967878818511963, -0.06396530568599701, 0.8968139290809631, 2.061156749725342, 0.04147633910179138, 0.016119062900543213] ,
                                [-0.8072128295898438, -1.761876106262207, 1.9285153150558472, -1.4094816446304321, -0.05636752024292946, 0.9160248637199402, 2.047523021697998, 0.03638796508312225, 0.016933633014559746] ,
                                [-0.7901577949523926, -1.7587257623672485, 1.9337682723999023, -1.4098329544067383, -0.04972277954220772, 0.9325187802314758, 2.0361878871917725, 0.019517341628670692, 0.012171330861747265]])
# waypoints are of length 29
THRESHOLD_MICROWAVE = -0.65 # if both qpos[22] < threshold, then compare gripper position, else compare qpos[22]

def get_next_target(state, task='microwave'):
    # return next target and distance to next target
    state = state[:9]
    if task == 'microwave':
        next_idx = -1; min_dist = 1e6
        for i, wp in enumerate(MICROWAVE_WAYPOINTS):
            dist = np.linalg.norm(state - wp)
            if min_dist > dist:
                min_dist = dist; next_idx = i 

        while min_dist < 1e-3:
            next_idx += 1
            min_dist = np.linalg.norm(state - wp[next_idx])
        
        return next_idx, min_dist
    else:
        raise NotImplementedError

# BEGIN_KETTLE = 

OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }
OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }