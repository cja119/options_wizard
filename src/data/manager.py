"""
Data manager class, to access data via SQL
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Optional
from pathlib import Path; import os
from dotenv import load_dotenv
import copy

import pandas as pd
import json

from concurrent.futures import ProcessPoolExecutor, as_completed

if TYPE_CHECKING:
    from src.universe import Universe

def make_hashable(obj):
    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(i) for i in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        return tuple(sorted(make_hashable(i) for i in obj))
    else:
        return obj
    
def _run_pipeline_for_tick(tick, methods, data_loader):
    try:
        data = data_loader(tick, return_data=True)
        first = True
        model_params = {}
        for kwargs, method in methods:
            kwargs['tick'] = tick
            if  hasattr(method, '_is_dataprep'):
                outputs = method(outputs, **kwargs)
            else:
                if kwargs.get('_source') == 'Transformer':
                    result = method(data, **kwargs)
                    data = result
                if kwargs.get('_source') == 'Features':
                    result = method(data, **kwargs)
                    if first:
                        outputs = result
                        first = False
                    else:
                        outputs = pd.concat([outputs, result], axis=1)
                if kwargs.get('_source') == 'Model':
                    name, params, result = method(outputs, **kwargs)
                    model_params[name] = params
                    outputs = pd.concat([outputs, result], axis=1)
                del result
        return tick, outputs, data, model_params
    except Exception as e:
        return tick, pd.DataFrame(), e, {}

def _run_single_method(tick, method, kwargs, data_dict, outputs_dict):
    """Run a single method for one tick and return updated data/output."""
    if method.__name__ == 'prepare_data':
        data = data_dict[tick]
        kwargs.append({'tick': tick})
        output = method(outputs_dict.get(tick, pd.DataFrame(index=data.index)), data, **kwargs)
        return tick, output
    data = data_dict[tick]
    return tick, method(data, **kwargs)

class SubProblemWrapper:
    def __init__(self, ticks: list[str] | str, manager: DataManager):
        self.ticks: list[str] = [ticks] if isinstance(ticks, str) else ticks
        self.manager: DataManager = manager
        return None

    def __call__(self, method=callable, **kwargs) -> None:
        """Applies a method to the data"""
        if self.manager.load_lazy:
            self.manager.add_method(self.ticks, method, kwargs)
        else:
            self.manager.run_method(self.ticks, method, kwargs)
        return None


class DataManager:

    def __init__(self, universe: Universe, load_lazy: bool = True):
        self.universe: Universe = universe
        self.load_lazy: bool = load_lazy
        self._method_pipeline: dict[str, list[tuple[dict, callable]]] = {}

        self.data: dict[str, pd.DataFrame] = {}
        self.outputs: dict[str, pd.DataFrame] = {}
        self.model_params: dict[str, dict] = {}
        self.combined_outputs: dict[str, pd.DataFrame] = {}

        if not self.load_lazy:
            self._load_data()

        return None

    def _load_data(self, ticks: list[str] | str | None = None, return_data: bool = False) -> pd.DataFrame:
        """Loads all data into memory"""
        load_dotenv()

        if ticks is None: ticks = self.universe.ticks
        if isinstance(ticks, str): ticks = [ticks]

        tick_path = os.getenv("TICK_PATH", "")

        for tick in ticks:
            tick_file = tick_path + f"/{tick}.parquet"
            data = pd.read_parquet(tick_file)
            if return_data:        
                return data.loc[:, ~data.columns.duplicated()]
            else:
                self.data[tick] = data.loc[:, ~data.columns.duplicated()]
        return None

    def __getitem__(self, ticks: list[str] | str):
        # Quick safety check, important if we are saving methods for later.
        assert all(
            tick in self.universe.ticks for tick in ticks
        ), "One or more ticks not in universe"
        return SubProblemWrapper(ticks, self)

    def __call__(self, *args):
        SubProblemWrapper(self.universe.ticks, self)(list(*args))
        return None

    def add_method(
        self, ticks: list[str], method: callable, kwargs: Optional[dict]
    ) -> None:
        """Adds a method to the pipeline for later execution"""
        for tick in ticks:
            if tick not in self._method_pipeline:
                self._method_pipeline[tick] = []
            self._method_pipeline[tick].append((kwargs, method))
        return None

    def run_method(self, ticks: list[str], method: callable, kwargs: Optional[dict]) -> None:
        """Run a single method in parallel across ticks."""
        with ProcessPoolExecutor() as executor:
            if kwargs.get('_source') in ['Transformer', 'Features']:
                futures = {
                    executor.submit(_run_single_method, tick, method, kwargs, self.data, self.outputs): tick
                    for tick in ticks
                }
            elif kwargs.get('_source') == 'Model':
                futures = {
                    executor.submit(_run_single_method, tick, method, kwargs, self.outputs, self.model_params): tick
                    for tick in ticks
                }

            for f in as_completed(futures):
                tick, output = f.result()
                if method.__name__ == 'prepare_data':
                    self.outputs[tick] = output
                else:
                    if kwargs.get('_source') == 'Transformer':
                        self.data[tick] = output
                    if kwargs.get('_source') == 'Features':
                        if self.outputs.get(tick) is None:
                            self.outputs[tick] = output
                        else:
                            self.outputs[tick] = pd.concat([self.outputs[tick], output], axis=1)
                    if kwargs.get('_source') == 'Model':
                        name, model_param = output
                        if tick not in self.model_params:
                            self.model_params[tick] = {}
                        self.model_params[tick][name] = model_param
        return None

    def test_pipeline(self, tick: str) -> pd.DataFrame:
        """Tests the pipeline for a single tick"""
        data = self._load_data([tick])
        df = data[tick]
        for kwargs, method in self._method_pipeline.get(tick, []):
            result = method(df, **kwargs)
            if kwargs.get('_source') in ['Transformer', 'Model']:
                df.loc[:, :] = result
            if kwargs.get('_source') == 'Features':
                pass  # For testing, we can ignore storing features
            del result
        return df

    def execute_pipeline(self, retain_data: bool = False, save_data: bool = False, n_workers: int = 8) -> None:
        """Executes all stored methods in the pipeline"""
        failed = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            posts = {}
            for tick, methods in self._method_pipeline.items():
                for kwargs, method in  methods.copy():
                    if kwargs.get('all_stocks', False):
                        print(f"\rDeferring method {method.__name__} for all stocks", end='', flush=True)
                        methods.remove((kwargs, method))
                        key = (make_hashable(kwargs), method)
                        if key in posts:
                            posts[key].append(tick)
                        else:
                            posts[key] = [tick]

            futures = {
                executor.submit(_run_pipeline_for_tick, tick, methods.copy(), copy.deepcopy(self._load_data)): tick
                for tick, methods in self._method_pipeline.items() 
            }
            for f in as_completed(futures):
                tick, output, data, model_params = f.result()
                if output.empty:
                    print(f"Analysis crashed for {tick}, reattempting, reason: {data}")
                    failed.append(tick)
                else:
                    self.outputs[tick] = output
                    self.model_params[tick] = model_params
                    if retain_data:
                        self.data[tick] = data
                    if save_data:
                        self.save_parquet(tick=tick, data=save_data)

        with ProcessPoolExecutor(max_workers=n_workers//2) as executor:
            futures = {
                executor.submit(_run_pipeline_for_tick, tick, methods, self._load_data): tick
                for tick, methods in self._method_pipeline.items() if tick in failed
            }
            for f in as_completed(futures):
                tick, output, data, model_params = f.result()
                if output.empty:
                    print(f"Analysis crashed twice for {tick}, removing from Universe. Reason {data}")
                    self.universe.ticks.remove(tick)
                else:
                    self.outputs[tick] = output
                    self.model_params[tick] = model_params
                    if retain_data:
                        self.data[tick] = data
                    if save_data:
                        self.save_parquet(tick=tick, data=save_data)

        for (kwargs_items, method), ticks in posts.items():
            kwargs = dict(kwargs_items)
            for tick in ticks:
                if tick not in self.universe.ticks:
                    ticks.remove(tick)
            for key, value in kwargs.items():
                if isinstance(value, tuple):
                    kwargs[key] = list(value)
            print(f"Running post-processing method {method.__name__} for ticks {ticks}")
            print(kwargs)
            self.run_method_combined(ticks, method, kwargs)

        return None

    def run_method_combined(self, ticks, method, kwargs) -> None:
        """Run a single method for multiple ticks together."""
        name = ''
        for tick in ticks:
            self.outputs[tick]['tick'] = tick
            name += tick + '_'
        kwargs['tick'] = name[:-1]
        input_df = pd.concat([self.outputs[tick] for tick in ticks], axis=0)
        name, params, preds = method(input_df, **kwargs)
        output_df = pd.concat([input_df, preds], axis=1)
        if method.__name__ not in self.combined_outputs:
            self.combined_outputs[method.__name__] = output_df
        else:
            self.combined_outputs[method.__name__] = pd.concat(
                [self.combined_outputs[method.__name__], output_df], axis=0
            ).dropna()

        for tick in ticks:
            tick_df = output_df[output_df['tick'] == tick].drop(columns=['tick'])
            self.outputs[tick] = tick_df
            if tick not in self.model_params:
                self.model_params[tick] = {}
            self.model_params[tick][name] = params

        return None

    def save_parquet(self, path: Optional[str] = None, data: Optional[bool] = None,
                     model: Optional[bool] = None, tick: Optional[str] = None) -> None:
        """Saves the output data to a file"""
        if path is None:
            path = "./tmp/parquet"
        if Path(path).exists() is False:
            Path(path).mkdir(parents=True, exist_ok=True)

        if tick is not None:
            self.outputs[tick].to_parquet(f"{path}/{tick}_output.parquet")
            if data:
                self.data[tick].to_parquet(f"{path}/{tick}_data.parquet")
            return None
        for tick, output in self.outputs.items():
            output.to_parquet(f"{path}/{tick}_output.parquet")
            if data:
                self.data[tick].to_parquet(f"{path}/{tick}_data.parquet")
        if model: 
            with open(f"{path}/model_params.json", "w") as f:
                json.dump(self.model_params, f)
        return None

    def load_parquet(self, tick: Optional[str] = None, path: Optional[str] = None,
                     model: Optional[bool] = None, data: Optional[bool] = None) -> None:
        """Loads output data from a file"""
        if path is None:
            path = "./tmp/parquet"
        if tick is not None:
            file_path = f"{path}/{tick}_output.parquet"
            if Path(file_path).exists():
                self.outputs[tick] = pd.read_parquet(file_path)
            else:
                self.outputs[tick] = pd.DataFrame()
            if data:
                data_file_path = f"{path}/{tick}_data.parquet"
                if Path(data_file_path).exists():
                    self.data[tick] = pd.read_parquet(data_file_path)
                else:
                    self.data[tick] = pd.DataFrame()
            if model:
                model_file_path = f"{path}/model_params.json"
                if Path(model_file_path).exists():
                    with open(model_file_path, "r") as f:
                        self.model_params = json.load(f)
                else:
                    self.model_params = {}
        else:
            for tick in self.universe.ticks:
                file_path = f"{path}/{tick}_output.parquet"
                if Path(file_path).exists():
                    self.outputs[tick] = pd.read_parquet(file_path)
                else:
                    self.outputs[tick] = pd.DataFrame()
                if data:
                    data_file_path = f"{path}/{tick}_data.parquet"
                    if Path(data_file_path).exists():
                        self.data[tick] = pd.read_parquet(data_file_path)
                    else:
                        self.data[tick] = pd.DataFrame()

                if model:
                    model_file_path = f"{path}/model_params.json"
                    if Path(model_file_path).exists():
                        with open(model_file_path, "r") as f:
                            self.model_params = json.load(f)
                    else:
                        self.model_params = {}   
        return None
