import argparse
import json
import sys
from .predictor import DEFAULT,Predictor,read


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default=DEFAULT)
    subs=parser.add_subparsers(dest="cmd",required=True)
    subs.add_parser("metadata")
    for k in ("predict","validate","predict-batch"): subs.add_parser(k).add_argument("input")
    args=parser.parse_args(argv)
    try:
        p=Predictor(args.config)
        if args.cmd=="metadata": result=p.describe()
        else:
            payload=read(args.input)
            if args.cmd=="validate": result={"warnings":p.validate(payload)[1],"valid":True}
            elif args.cmd=="predict-batch": result=p.predict_batch(payload)
            else: result=p.predict_record(payload)
        print(json.dumps(result,ensure_ascii=False))
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(str(exc),file=sys.stderr)
        return 2


if __name__=="__main__": sys.exit(main())
