from profile import BaseProfile
from settings import *
import re

class Profile(BaseProfile):
    def start(self):
        with open(pjoin(PATH['test_data'], 'test.in'), 'w') as fw, \
             open(pjoin(PATH['data'], 'testing_data.txt')) as f, \
             open(pjoin(PATH['data'], 'correct', 'MSR_Sentence_Completion_Challenge_V1',
                 'Data', 'Holmes.machine_format.answers.txt')) as fans:
            #regex = re.compile(r'^(.*)\[([^]])\](.*)$')
            regex = re.compile(r'^(\d+[a-e]\)) +(.*)\[([^]]*)\](.*)$')
            while True:
                s = f.readline().strip('\n')
                if not s: return 

                fin_s = regex.sub(r'\2[]\4', s)
                match = regex.match(s)
                opt = [match.group(3)]
                for i in range(4):
                    s = f.readline()
                    match = regex.match(s)
                    opt.append(match.group(3))
                sans = fans.readline().strip('\n')
                ans = regex.match(sans).group(3)
                fw.write('{}\n{}\n{} {}\n'.format(
                    fin_s, ' '.join(opt), ans, opt.index(ans)
                ))


