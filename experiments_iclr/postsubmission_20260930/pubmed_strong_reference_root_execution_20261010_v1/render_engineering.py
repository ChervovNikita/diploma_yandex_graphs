"""Render three disabled-source engineering interfaces after actual root review."""
import json
from render_common import arguments,context,release,plan


def main():
    args=arguments(__doc__);ctx=context(args)
    rows=[release('ENGINEERING_'+c+'.json','engineering',ctx)
          for c in ('single_native','single_mean4_dropout','independent4_own')]
    result=plan('engineering',rows,ctx)
    print(json.dumps(dict(engineering_interfaces=3,owner_plan=result,science_not_started=True,engineering_not_started=True)))


if __name__=='__main__':main()
