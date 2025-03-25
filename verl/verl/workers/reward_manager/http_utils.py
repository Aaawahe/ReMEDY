import asyncio
import aiohttp
from aiohttp import ClientSession
import requests
import json

async def async_do_request(
    url: str,
    data,
    timeout=10,
):
    async with ClientSession() as session:
        async with session.post(
            url,
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=timeout,

        ) as response:
            try:
                res = await response.json()
                return res
            except Exception as e:
                print(f"####### do request error: {e}")
                return e

session = requests.Session()
session.mount('http://', requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=100))
session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=100))

def do_request(
    url: str,
    data,
    timeout=10,
):
    global session
    try:
        res = session.post(
            url,
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=timeout
        )
        return res.json()
    except Exception as e:
        print(f"####### do request error: {e}")
        return e


def do_parallel_request(params, datas, num_processes=8):

    try:
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for data in datas:
                futures.append(executor.submit(
                    do_request,
                    data=data,
                    **params
                ))
            
            results = [future.result() for future in futures]
    
    except Exception as e:
        print(f"####### do parallel request error: {e}")
        results = [None for _ in range(len(datas))]
    
    return results